def fibonacci(n):
    numbers = []
    a, b = 0, 1
    for _ in range(n):
        numbers.append(a)
        a, b = b, a + b
    return numbers

fib_numbers = fibonacci(10)
print("First 10 Fibonacci numbers:")
for i, num in enumerate(fib_numbers, 1):
    print(f"  F({i}) = {num}")
