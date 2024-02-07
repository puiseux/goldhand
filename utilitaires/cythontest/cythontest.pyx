# This is a sample Python script.

def factors(n, result):
    if n <= 1:
        return result
    for i in range(2, n + 1):
        if n % i == 0:
            result.append(i)
            return factors(n // i, result)

def main():
    numbers = range(1, 50001)

    for n in numbers:
        factors(n, [1])

if __name__ == '__main__':
    main()
