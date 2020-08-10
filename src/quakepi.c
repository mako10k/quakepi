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
#include <assert.h>
#include <pthread.h>

#include <fftw3.h>

#define FFT_LOG2 9
#define FFT_SIZE (1<<(FFT_LOG2))
#define FFT_HALF_SIZE (1<<(FFT_LOG2 - 1))
#define SAMPLE_PERIOD 0.01

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
  struct i2c_msg msgs[] = {
    {
     .addr = addr,
     .flags = 0,
     .len = 1,
     .buf = (void *) &reg,
     },
    {
     .addr = addr,
     .flags = 0,
     .len = len,
     .buf = (uint8_t *) buf,
     }
  };
  struct i2c_rdwr_ioctl_data data = { msgs, 2 };
  return ioctl (fd, I2C_RDWR, &data);
}

static double *
magnitude_filter_fr (double sample_period, size_t samples)
{
  double *filter_fr = malloc (sizeof (double) * samples);
  filter_fr[0] = 0;
  for (int i = 1; i < samples; i++)
    {
      double fq = ((double) i / samples) / sample_period;
      double X2 = (fq / 10) * (fq / 10);
      double fh =
        1 / sqrt (1 +
                  (0.694 +
                   (0.241 +
                    (0.0557 +
                     (0.009664 +
                      (0.00134 + 0.00155 * X2) * X2) * X2) * X2) * X2) * X2);
      double l3 = (fq / 0.5) * (fq / 0.5) * (fq / 0.5);
      double fl = sqrt (1 - exp (-l3));
      double fc = sqrt (100.0 / i);
      filter_fr[i] = fh * fl * fc / samples;
    }
  return filter_fr;
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static void *
magnitude_thread (void *arg)
{
  int16_t *p_from = arg;
  int16_t p[FFT_HALF_SIZE * 3];
  fftw_complex t[3][FFT_SIZE];
  fftw_complex f[FFT_SIZE];
  fftw_complex u[3][FFT_SIZE];
  double *frs;
  double m[FFT_SIZE];
  int count = 0;

  // 周波数データの係数
  // 直流分(重力)カット
  // FFTの正規化係数 N/2
  // 周期効果 f**(-1/2)
  // ハイカット (1+0.694*X**2+0.24*X**4+0.0557*X**6+0.009664*X**8+0.00134*X**10+0.000155*X**12)**(-1/2)
  //   X=f/fhc (fhc = 10Hz)
  // ローカット (1-exp(-(f/flc)**3))**(1/2)
  //   hlc=0.5Hz
  frs = magnitude_filter_fr (SAMPLE_PERIOD, FFT_SIZE);

  fftw_plan plan_fwd[3];
  fftw_plan plan_bck[3];
  plan_fwd[0] =
    fftw_plan_dft_1d (FFT_SIZE, t[0], f, FFTW_FORWARD, FFTW_ESTIMATE);
  plan_fwd[1] =
    fftw_plan_dft_1d (FFT_SIZE, t[1], f, FFTW_FORWARD, FFTW_ESTIMATE);
  plan_fwd[2] =
    fftw_plan_dft_1d (FFT_SIZE, t[2], f, FFTW_FORWARD, FFTW_ESTIMATE);
  plan_bck[0] =
    fftw_plan_dft_1d (FFT_SIZE, f, u[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  plan_bck[1] =
    fftw_plan_dft_1d (FFT_SIZE, f, u[1], FFTW_BACKWARD, FFTW_ESTIMATE);
  plan_bck[2] =
    fftw_plan_dft_1d (FFT_SIZE, f, u[2], FFTW_BACKWARD, FFTW_ESTIMATE);
  while (true)
    {
      time_t now;
      memmove (t[0], t[0] + FFT_HALF_SIZE,
               sizeof (fftw_complex) * FFT_HALF_SIZE);
      memmove (t[1], t[1] + FFT_HALF_SIZE,
               sizeof (fftw_complex) * FFT_HALF_SIZE);
      memmove (t[2], t[2] + FFT_HALF_SIZE,
               sizeof (fftw_complex) * FFT_HALF_SIZE);
      // Copy Half sample data
      pthread_mutex_lock (&mutex);
      pthread_cond_wait (&cond, &mutex);
      now = time (NULL);
      memcpy (p, p_from, sizeof (p));
      pthread_mutex_unlock (&mutex);

      // set real part of sample data
      for (int i = 0; i < FFT_HALF_SIZE; i++)
        {
          t[0][i + FFT_HALF_SIZE][0] = p[i * 3 + 0] * (980.665 / 16384.0);
          t[0][i + FFT_HALF_SIZE][1] = 0;
          t[1][i + FFT_HALF_SIZE][0] = p[i * 3 + 1] * (980.665 / 16384.0);
          t[1][i + FFT_HALF_SIZE][1] = 0;
          t[2][i + FFT_HALF_SIZE][0] = p[i * 3 + 2] * (980.665 / 16384.0);
          t[2][i + FFT_HALF_SIZE][1] = 0;
        }
      if (count++ == 0)
        continue;

      // calculate for each axis
      for (int i = 0; i < 3; i++)
        {
          // Do FFT
          fftw_execute (plan_fwd[i]);

          // Apply filter
          for (int j = 0; j < FFT_SIZE; j++)
            {
              f[j][0] *= frs[j];
              f[j][1] *= frs[j];
            }

          // Do InvFFT
          fftw_execute (plan_bck[i]);
        }
      // convert to scalar accel
      for (int i = 0; i < FFT_SIZE; i++)
        m[i] =
          sqrt (u[0][i][0] * u[0][i][0] + u[1][i][0] * u[1][i][0] +
                u[2][i][0] * u[2][i][0]);

      // sort desc
      int dcomp (const void *a, const void *b)
      {
        return *(double *) a > *(double *) b ? -1 : 1;
      }
      qsort (m, FFT_SIZE, sizeof (double), dcomp);

      // select max 0.3sec accel
      int idx = 0.3 / SAMPLE_PERIOD;

      // convert accel to magnitude
      double magnitude = 2 * log10 (m[idx]) + 0.94;
      char date[64];
      strftime (date, sizeof (date), "%Y-%m-%d %H:%M:%S", localtime (&now));
      printf ("%s,%4.2lf,%6.2lf\n", date, magnitude, m[idx]);
    }
  return NULL;
}

int
main (int argc, char *argv[])
{
  int tfd = timerfd_create (CLOCK_REALTIME, 0);
  int fd = qp_i2c_open (1);
  uint8_t cntl = 0;
  uint16_t data[3];
  int idx = 0;
  struct itimerspec its;
  int16_t p[3 * FFT_HALF_SIZE];
  int16_t p_to[3 * FFT_HALF_SIZE];
  pthread_t tid;
  its.it_value.tv_sec = SAMPLE_PERIOD;
  its.it_value.tv_nsec = (SAMPLE_PERIOD - its.it_value.tv_sec) * 1000000000;
  its.it_interval = its.it_value;
  timerfd_settime (tfd, 0, &its, NULL);
  qp_i2c_write (fd, 0x68, 0x6B, &cntl, sizeof (cntl));

  pthread_create (&tid, NULL, magnitude_thread, p_to);
  while (true)
    {
      uint64_t cnt;
      read (tfd, &cnt, sizeof (cnt));
      qp_i2c_read (fd, 0x68, 0x3B, (uint8_t *) data, sizeof (data));
      p[idx++] = be16toh (data[0]);
      p[idx++] = be16toh (data[1]);
      p[idx++] = be16toh (data[2]);
      if (idx == FFT_HALF_SIZE * 3)
        {
          pthread_mutex_lock (&mutex);
          memcpy (p_to, p, sizeof (p_to));
          pthread_mutex_unlock (&mutex);
          pthread_cond_signal (&cond);
          idx = 0;
        }
    }
  exit (EXIT_SUCCESS);
}
