# -*- coding: utf-8 -*-

# 1.Import the numpy package under the name np (★☆☆)
import numpy as np

# 2.Print the numpy version and the configuration (★☆☆)
print(np.__version__)

# 3. Create a zero vector of size 10 (★☆☆)
np.zeros(shape=[10])

# 4. How to find the memory size of any array (★☆☆)
x = np.zeros(shape=[10], dtype=np.float32)
print(x.size)
print(x.itemsize)

# 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
print(np.info(np.add))

# 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
x = np.zeros(shape=[10])
x[4] = 1

# 7. Create a vector with values ranging from 10 to 49 (★☆☆)
np.arange(start=10, stop=50, step=1)

# 8. Reverse a vector (first element becomes last) (★☆☆)
x = np.arange(start=10, stop=50, step=1)
np.flip(x, axis=-1)
x[::-1]

# 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
x = np.arange(0, 9)
x.reshape(3, 3)

# 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
x = np.array([1, 2, 0, 0, 4, 0])
np.where(x != 0)
np.nonzero(x)

# 11. Create a 3x3 identity matrix (★☆☆)
np.eye(3)

# 12. Create a 3x3x3 array with random values (★☆☆)
np.random.random((3, 3, 3))
np.random.randint(27, size=(3, 3, 3))

# 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
x = np.random.random((10, 10))
x.max(), x.min()

# 14. Create a random vector of size 30 and find the mean value (★☆☆)
x = np.random.random(30)
x.mean()

# 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
x = np.ones((4, 4))
x[1:-1, 1:-1] = 0

# 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
x = np.ones((4, 4))
y = np.pad(array=x, pad_width=1, mode='constant')

# 17. What is the result of the following expression? (★☆☆)
0 * np.nan           # nan
np.nan == np.nan     # False
np.inf > np.nan      # False
np.nan - np.nan      # nan
0.3 == 3 * 0.1       # False

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
x = np.array([1, 2, 3, 4])
np.diag(x)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
x = np.zeros((8, 8))
x[1::2, ::2] = 1
x[::2, 1::2] = 1

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
np.unravel_index(100, (6, 7, 8))

# 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
np.tile(np.array([[0, 1], [1, 0]]), (4, 4))

# 22. Normalize a 5x5 random matrix (★☆☆)
x = np.random.random(size=(5, 5))
x_mean = np.mean(x)
x_var = np.var(x)
x_norm = (x - x_mean) / np.sqrt(x_var)
# code you have written above in 22nd question is called standardization
# code for the actual normalization
x = np.random.random(size=(5, 5))
mx = x.max()
mn = x.min()
x_norm = (x - mn) / (mx - mn)

# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
x = np.arange(15).reshape((5, 3))
y = np.arange(6).reshape((3, 2))
np.dot(x, y)

# 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
x = np.arange(11)
x[(x >= 3) & (x <= 8)] = -1

# 26. What is the output of the following script? (★☆☆)
print(sum(range(5), -1))  # 9
from numpy import *

print(sum(range(5), -1))  # 10

# 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
Z = np.arange(3)
Z ** Z       # = [0^0, 1^1, 2^2] = [1, 1, 4]
2 << Z >> 2  # = [0, 1, 2]
Z < - Z      # = [False, False, False]
1j * Z       # = [0 + 0.j, 0 + 1.j, 0 + 2.j]
Z / 1 / 1    # = [0, 1, 2]
Z < Z > Z    # ValueError

# 28. What are the result of the following expressions? (★☆☆)
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

# 29. How to round away from zero a float array ? (★☆☆)
x = np.random.uniform(-10, +10, 10)
np.copysign(np.ceil(np.abs(x)), x)

# 30. How to find common values between two arrays? (★☆☆)
x = np.arange(0, 10)
y = np.arange(5, 15)
np.intersect1d(x, y)

# 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# 32. Is the following expressions true? (★☆☆)
print(np.sqrt(-1) == np.emath.sqrt(-1))  # False

# 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
today = np.datetime64('today', 'D')
yesterday = today - np.datetime64(1, 'D')
tomorrow = today + np.datetime64(1, 'D')

# 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
x = np.arange('2016-07', '2016-08', dtype='datetime64[D]')

# 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)

# 36. Extract the integer part of a random array using 5 different methods (★★☆)
x = np.random.uniform(-10, +10, 10)
x.astype(np.int32)
np.trunc(x)

# 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
x = np.zeros((5, 5))
x += np.arange(5)

# 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
def generate_integers():
    for i in range(10):
        yield i
np.fromiter(generate_integers(), dtype=np.float32, count=-1)

# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
np.linspace(start=0, stop=11, num=10, endpoint=False)

# 40. Create a random vector of size 10 and sort it (★★☆)
x = np.random.random(size=10)
x.sort()

# 41. How to sum a small array faster than np.sum? (★★☆)
import time

begin_time = time.time()
x = np.arange(1000000)
np.sum(x)
end_time = time.time()
print(str(1000*(end_time - begin_time)))

begin_time2 = time.time()
x = np.arange(1000000)
np.add.reduce(x)
end_time2 = time.time()
print(str(1000*(end_time2 - begin_time2)))

# 42. Consider two random array A and B, check if they are equal (★★☆)
A = np.arange(10)
B = np.arange(10)
np.array_equal(A, B)

# 43. Make an array immutable (read-only) (★★☆)
x = np.zeros(10)
x.flags.writeable = False
x[1] = 1   # ValueError: assignment destination is read-only

# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
Z = np.random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X ** 2 + Y ** 2)
T = np.arctan2(Y, X)
print(R)
print(T)

# 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
x = np.random.random(10)
x[x == np.max(x)] = 0
x[x.argmax()] = 0

# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)
z = np.zeros((5, 5), [('x', float), ('y', float)])
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))

# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
x = np.random.random((5, 5))
y = np.random.random((5, 5))
c = 1 / (x - y)
c = 1 / np.subtract.outer(x, y)
np.linalg.det(c)

# 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


# 49. How to print all the values of an array? (★★☆)

# 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

