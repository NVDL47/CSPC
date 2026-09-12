import time
from decay import simulate_loop, simulate

N0 = 200000
lam = 0.4

start1 = time.perf_counter()
simulate_loop(N0, lam)
end1 = time.perf_counter()
myTime1 = end1 - start1

start2 = time.perf_counter()
simulate(N0, lam)
end2 = time.perf_counter()
myTime2 = end2 - start2

difference = myTime1 / myTime2

print(f"Pure Python: {myTime1}")
print(f"NumPy: {myTime2}")
print(f"NumPy is {difference} times faster.")