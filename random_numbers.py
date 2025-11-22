import random

# Generate and print 10 random numbers between 1 and 100
print("10 random numbers between 1 and 100:")
for i in range(10):
    random_number = random.randint(1, 100)
    print(f"Number {i+1}: {random_number}")