#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/timerfd.h>
#include <sys/time.h>
#include <endian.h>
#include <math.h>
#include <stdbool.h>
#include <signal.h>
#include <string.h>
#include <pthread.h>

#include <complex.h>
#include <fftw3.h>

#if HAVE_QSORTRANK_H
#include <qsortrank.h>
#endif

static int
qp_i2c_open (int bus)
{
  char i2cdev[] = "/dev/i2c-0";
  i2cdev[9] += bus;
  return open (i2cdev, O_RDWR);
}

static int
qp_i2c_read (int fd, int addr, int reg, uint8_t * buf, size_t len)
{
  uint8_t regbuf = reg;
  struct i2c_msg msgs[] = {
    {
     .addr = addr,
     .flags = 0,
     .len = 1,
     .buf = &regbuf,
     },
    {
     .addr = addr,
     .flags = I2C_M_RD,
     .len = len,
     .buf = buf,
     }
  };
  struct i2c_rdwr_ioctl_data data = { msgs, 2 };
  return ioctl (fd, I2C_RDWR, &data);
}

static int
qp_i2c_write (int fd, int addr, int reg, const uint8_t * buf, size_t len)
{
  uint8_t wbuf[len + 1];
  wbuf[0] = reg;
  memcpy (wbuf + 1, buf, len);
  struct i2c_msg msgs[] = {
    {
     .addr = addr,
     .flags = 0,
     .len = len + 1,
     .buf = wbuf,
     }
  };
  struct i2c_rdwr_ioctl_data data = { msgs, 1 };
  return ioctl (fd, I2C_RDWR, &data);
}

// 直流分(重力)カット
// FFTの正規化係数 N/2
// 周期効果 f**(-1/2)
// ハイカット (1+0.694*X**2+0.24*X**4+0.0557*X**6+0.009664*X**8+0.00134*X**10+0.000155*X**12)**(-1/2)
//   X=f/fhc (fhc = 10Hz)
// ローカット (1-exp(-(f/flc)**3))**(1/2)
//   hlc=0.5Hz

#define HIGH_CUT_FREQ 10.0
#define LOW_CUT_FREQ 0.5
static double *
magnitude_filter_fr (double sample_period, size_t samples)
{
  double *filter_fr = malloc (sizeof (double) * (samples / 2 + 1));
  filter_fr[0] = 0;
  for (int i = 1; i < (samples / 2 + 1); i++)
    {
      double fq = ((double) i / samples) / sample_period;
      double X2 = (fq / HIGH_CUT_FREQ) * (fq / HIGH_CUT_FREQ);
      double fh =
        1 / sqrt (1 +
                  (0.694 +
                   (0.241 +
                    (0.0557 +
                     (0.009664 +
                      (0.00134 + 0.00155 * X2) * X2) * X2) * X2) * X2) * X2);
      double l3 =
        (fq / LOW_CUT_FREQ) * (fq / LOW_CUT_FREQ) * (fq / LOW_CUT_FREQ);
      double fl = sqrt (1 - exp (-l3));
      double fc = 1 / sqrt (fq);
      filter_fr[i] = fh * fl * fc / samples;
    }
  return filter_fr;
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static void
print_usage (FILE * fp, int argc, char *argv[])
{
  fprintf (fp, "\n");
  fprintf (fp, "Usage:\n");
  fprintf (fp, "  %s [options]\n", argv[0]);
  fprintf (fp, "\n");
  fprintf (fp, "Options:\n");
  fprintf (fp,
           "  -n samples : samples to filter in log2 scale (DEFAULT: 8 (=256))\n");
  fprintf (fp,
           "  -p period  : output period in seconds        (DEFAULT: 2.0))\n");
  fprintf (fp, "  -v         : print verbose output\n");
  fprintf (fp, "  -h         : print this usage\n");
  fprintf (fp, "\n");
}

struct thread_args
{
  int16_t *p;
  size_t samples;
  double sampling_period;
};

static void *
magnitude_thread (void *arg)
{
  struct thread_args *args = arg;
  size_t fft_half_size = args->samples / 2;
  size_t fft_freq_size = fft_half_size + 1;
  int16_t p[fft_half_size * 3];
  double t[3][args->samples];
  fftw_complex f[fft_freq_size];
  double u[3][args->samples];
  double *frs;
  double m[args->samples];
  int count = 0;

  // 周波数データの係数
  frs = magnitude_filter_fr (args->sampling_period, args->samples);

  fftw_plan plan_fwd[3];
  fftw_plan plan_bck[3];
  plan_fwd[0] = fftw_plan_dft_r2c_1d (args->samples, t[0], f, FFTW_ESTIMATE);
  plan_fwd[1] = fftw_plan_dft_r2c_1d (args->samples, t[1], f, FFTW_ESTIMATE);
  plan_fwd[2] = fftw_plan_dft_r2c_1d (args->samples, t[2], f, FFTW_ESTIMATE);
  plan_bck[0] = fftw_plan_dft_c2r_1d (args->samples, f, u[0], FFTW_ESTIMATE);
  plan_bck[1] = fftw_plan_dft_c2r_1d (args->samples, f, u[1], FFTW_ESTIMATE);
  plan_bck[2] = fftw_plan_dft_c2r_1d (args->samples, f, u[2], FFTW_ESTIMATE);
  while (true)
    {
      time_t now;
      memcpy (t[0], t[0] + fft_half_size, sizeof (*t[0]) * fft_half_size);
      memcpy (t[1], t[1] + fft_half_size, sizeof (*t[1]) * fft_half_size);
      memcpy (t[2], t[2] + fft_half_size, sizeof (*t[2]) * fft_half_size);
      // Copy Half sample data
      pthread_mutex_lock (&mutex);
      pthread_cond_wait (&cond, &mutex);
      now = time (NULL);
      memcpy (p, args->p, sizeof (p));
      pthread_mutex_unlock (&mutex);

      // set real part of sample data
      for (int i = 0; i < fft_half_size; i++)
        {
          t[0][i + fft_half_size] = p[i * 3 + 0] * (980.665 / 16384.0);
          t[1][i + fft_half_size] = p[i * 3 + 1] * (980.665 / 16384.0);
          t[2][i + fft_half_size] = p[i * 3 + 2] * (980.665 / 16384.0);
        }
      if (count++ == 0)
        continue;

      // calculate for each axis
      for (int i = 0; i < 3; i++)
        {
          // Do FFT
          fftw_execute (plan_fwd[i]);

          // Apply filter
          for (int j = 0; j < fft_freq_size; j++)
            f[j] *= frs[j];

          // Do InvFFT
          fftw_execute (plan_bck[i]);
        }
      // convert to scalar accel
      for (int i = 0; i < args->samples; i++)
        m[i] =
          sqrt (u[0][i] * u[0][i] + u[1][i] * u[1][i] + u[2][i] * u[2][i]);

      // sort desc
      int dcomp (const void *a, const void *b)
      {
        return *(double *) a > *(double *) b ? -1 : 1;
      }
      // select max 0.3sec accel
      int idx = 0.3 / args->sampling_period;

#if HAVE_QSORTRANK_H
      qsortrank (m, args->samples, sizeof (double), idx, dcomp);
#else
      qsort (m, args->samples, sizeof (double), dcomp);
#endif

      // convert accel to magnitude
      double magnitude = 2 * log10 (m[idx]) + 0.94;
      char date[64];
      strftime (date, sizeof (date), "%Y-%m-%d %H:%M:%S", localtime (&now));
      magnitude = floor (round (magnitude * 100) / 10) / 10;
      printf ("%s,%.1lf,%.2lf\n", date, magnitude, m[idx]);
      fflush (stdout);
    }
  return NULL;
}

int
main (int argc, char *argv[])
{
  int opt;
  int opt_samples = 8;
  double opt_period = 2.0;
  int opt_verbose = 0;

  while ((opt = getopt (argc, argv, "n:p:vh")) != -1)
    {
      switch (opt)
        {
        case 'n':
          {
            char *p;
            opt_samples = strtol (optarg, &p, 0);
            if (optarg == p || *p)
              {
                fprintf (stderr, "invalid argument for -n\n");
                print_usage (stderr, argc, argv);
                exit (EXIT_FAILURE);
              }
          }
          if (opt_samples < 4 || opt_samples > 16)
            {
              fprintf (stderr, "-n must between 4 and 16\n");
              print_usage (stderr, argc, argv);
              exit (EXIT_FAILURE);
            }
          break;
        case 'p':
          {
            char *p;
            opt_period = strtod (optarg, &p);
            if (optarg == p || *p)
              {
                fprintf (stderr, "invalid argument for -p\n");
                print_usage (stderr, argc, argv);
                exit (EXIT_FAILURE);
              }
          }
          if (opt_period < 0.5 || opt_period > 10)
            {
              fprintf (stderr, "-p must between 0.5 and 10\n");
              print_usage (stderr, argc, argv);
              exit (EXIT_FAILURE);
            }
          break;
        case 'v':
          opt_verbose++;
          break;
        case 'h':
          print_usage (stdout, argc, argv);
          exit (EXIT_SUCCESS);
        }
    }
  if (optind < argc)
    {
      fprintf (stderr, "extra arguments\n");
      print_usage (stderr, argc, argv);
      exit (EXIT_FAILURE);
    }

  size_t fft_size = 1 << opt_samples;
  size_t fft_half_size = fft_size / 2;
  double sampling_period = opt_period / fft_half_size;
  if (opt_verbose)
    {
      fprintf (stderr, "FFT SIZE: %zu\n", fft_size);
      fprintf (stderr, "SAMPLING: every %lf sec / %lf Hz\n", sampling_period,
               1 / sampling_period);
      fprintf (stderr, "CALCULATE EVERY: %zu samples / %lf seconds\n",
               fft_half_size, opt_period);
      fprintf (stderr, "FILTER: resolution %lf Hz, max %lf Hz\n",
               1 / sampling_period / fft_size, 0.5 / sampling_period);
      fprintf (stderr, "RANK INDEX: %d of %d\n",
               (int) (0.3 / sampling_period), fft_size);
    }
  int tfd = timerfd_create (CLOCK_REALTIME, 0);
  int fd = qp_i2c_open (1);
  uint8_t cntl = 0;
  int lost_samples = 0;
  uint16_t data[3];
  int idx = 0;
  struct itimerspec its;
  int16_t p[3 * fft_half_size];
  int16_t p_to[3 * fft_half_size];
  pthread_t tid;
  its.it_value.tv_sec = sampling_period;
  its.it_value.tv_nsec = (sampling_period - its.it_value.tv_sec) * 1000000000;
  its.it_interval = its.it_value;
  timerfd_settime (tfd, 0, &its, NULL);
  qp_i2c_write (fd, 0x68, 0x1C, &cntl, sizeof (cntl));
  qp_i2c_write (fd, 0x68, 0x6B, &cntl, sizeof (cntl));

  struct thread_args args = {
    .p = p_to,
    .samples = fft_size,
    .sampling_period = sampling_period,
  };
  pthread_create (&tid, NULL, magnitude_thread, &args);
  while (true)
    {
      uint64_t cnt;
      read (tfd, &cnt, sizeof (cnt));
      lost_samples += cnt - 1;
      qp_i2c_read (fd, 0x68, 0x3B, (uint8_t *) data, sizeof (data));
      p[idx++] = be16toh (data[0]);
      p[idx++] = be16toh (data[1]);
      p[idx++] = be16toh (data[2]);
      if (idx == fft_half_size * 3)
        {
          pthread_mutex_lock (&mutex);
          memcpy (p_to, p, sizeof (p_to));
          if (opt_verbose && lost_samples)
            fprintf (stderr, "lost samples %d times, please change options\n",
                     lost_samples);
          pthread_mutex_unlock (&mutex);
          pthread_cond_signal (&cond);
          idx = 0;
          lost_samples = 0;
        }
    }
  exit (EXIT_SUCCESS);
}
