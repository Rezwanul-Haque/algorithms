import sys
import time


def bit_string(number: int, string: str) -> list:
#     Base cases -> when this recursive calls break
    if number == 1:
        return string
#     Recursuve cases
    else:
        return [digit + bits for digit in bit_string(1, string) for bits in bit_string(number - 1, string)]
    
   

number = 3
string = 'abc'
start_time = time.time()
print(bit_string(number, string))
total_time = time.time() - start_time
print(total_time)