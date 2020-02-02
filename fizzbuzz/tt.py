def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


print(fizz_buzz_decode(3, 0))
print(fizz_buzz_decode(3, 1))
print(fizz_buzz_decode(3, 2))
print(fizz_buzz_decode(3, 3))