import random
import string

# Generate and print 30 lines of random characters
print("30 lines of random characters:")
for i in range(30):
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"Line {i+1}: {random_chars}")