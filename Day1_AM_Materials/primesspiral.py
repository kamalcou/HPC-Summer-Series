import matplotlib.pyplot as plt
import numpy as np

# Check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

# Generate spiral coordinates
def generate_spiral(n_points, spacing=0.5):
    theta = np.linspace(0, n_points * spacing, n_points)
    r = theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Number of points to plot
N = 10000

# Generate spiral
x, y = generate_spiral(N)

# Plot
plt.figure(figsize=(8, 8))
for i in range(N):
    if is_prime(i):
        plt.plot(x[i], y[i], 'ro', markersize=3)  # red dot for prime
    else:
        plt.plot(x[i], y[i], 'k.', markersize=1)  # small black dot for non-prime

plt.axis('equal')
plt.axis('off')
plt.title("Prime Numbers in a Polar Spiral", fontsize=14)
plt.savefig('spiral.png')
plt.close()

# Count prime numbers between 0 and 10000
prime_count = sum(1 for i in range(10001) if is_prime(i))
print(f"Found {prime_count} prime numbers between 0 and 10000.")


