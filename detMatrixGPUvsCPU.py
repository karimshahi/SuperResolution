# وارد کردن کتابخانه‌های مورد نیاز
import tensorflow as tf
import numpy as np
import time

# تعریف یک تابع برای محاسبه دترمینان یک ماتریس با استفاده از یک دستگاه خاص
def compute_determinant(matrix, device):
  # انتخاب دستگاه
  with tf.device(device):
    # تبدیل ماتریس به یک تانسور
    tensor = tf.convert_to_tensor(matrix)
    # محاسبه دترمینان با استفاده از تابع det
    determinant = tf.linalg.det(tensor)
    # چاپ نتیجه
    print(f"The determinant of the matrix on {device} is {determinant}")

# ایجاد یک ماتریس 100 در 100 با اعداد تصادفی
matrix = np.random.rand(100, 100)

# محاسبه دترمینان ماتریس با استفاده از GPU های موجود
# شماره GPU ها را بر اساس سیستم خود تنظیم کنید
# اینجا فرض شده است که دو GPU با شماره 0 و 1 وجود دارند
gpus = ['/GPU:0', '/GPU:1']
for gpu in gpus:
  # ثبت زمان شروع
  start = time.time()
  # فراخوانی تابع محاسبه دترمینان
  compute_determinant(matrix, gpu)
  # ثبت زمان پایان
  end = time.time()
  # محاسبه مدت زمان اجرا
  duration = end - start
  # چاپ مدت زمان اجرا
  print(f"The execution time on {gpu} was {duration} seconds")

# محاسبه دترمینان ماتریس با استفاده از CPU
# ثبت زمان شروع
start = time.time()
# فراخوانی تابع محاسبه دترمینان
compute_determinant(matrix, '/CPU:0')
# ثبت زمان پایان
end = time.time()
# محاسبه مدت زمان اجرا
duration = end - start
# چاپ مدت زمان اجرا
print(f"The execution time on /CPU:0 was {duration} seconds")
