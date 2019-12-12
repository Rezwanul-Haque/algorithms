import sys
import time


def factorial_N(number: int) -> int:
#     Base cases -> when this recursive calls break
    if number == 0:
        return 1
#     Recursuve cases
    else:
        factorial = number * factorial_N(number - 1)
    
    print(factorial)
    return factorial
    
   

number = 4
start_time = time.time()
print("factorial: ", factorial_N(number))
total_time = time.time() - start_time
print(total_time)