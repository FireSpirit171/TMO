# Задача 1
s = input().strip() 
print("YES" if s.index("R") < s.index("M") else "NO")


# Задача 2
def find_three_powers_of_two(a):
    powers = []
    power = 1

    while power <= a:
        if power & a:
            powers.append(power)
        power <<= 1

    if len(powers) < 3:
        return -1

    return sum(sorted(powers, reverse=True)[:3])

n = int(input().strip())
results = []

for _ in range(n):
    a = int(input().strip())
    results.append(find_three_powers_of_two(a))

print("\n".join(map(str, results)))


# Задача 3
def adjust_values(total_days, required_good_days, values):
    lower_bound, upper_bound = values[0], values[1]
    total_adjustments = 0
    valid_days = 0
    
    for i in range(2, total_days):
        if lower_bound <= values[i] <= upper_bound:
            valid_days += 1
        else:
            if values[i] < lower_bound:
                total_adjustments += lower_bound - values[i]
            elif values[i] > upper_bound:
                total_adjustments += values[i] - upper_bound
            valid_days += 1
        
        if valid_days >= required_good_days:
            break
    
    return total_adjustments

total_days, required_good_days = map(int, input().split())
values = list(map(int, input().split()))
print(adjust_values(total_days, required_good_days, values))


# Задача 5
def calculate_min_cuts(n, s, a):
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    
    total = 0
    for l in range(1, n+1):
        for r in range(l, n+1):
            segment_sum = prefix[r] - prefix[l-1]
            cuts = (segment_sum + s - 1) // s
            total += cuts
    return total

n, s = map(int, input().split())
a = list(map(int, input().split()))
print(calculate_min_cuts(n, s, a))


# Задача 7
MOD = 998244353

def compute_powers(n, k, nums):
    from collections import defaultdict

    sum_pairs = defaultdict(int) 
    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = nums[i] + nums[j]
            sum_pairs[pair_sum] += 1

    output = []
    for r in range(1, k + 1):
        total_sum = 0
        for pair_sum, count in sum_pairs.items():
            total_sum += pow(pair_sum, r, MOD) * count
            total_sum %= MOD
        output.append(total_sum)

    return output

n, k = map(int, input().split())
nums = list(map(int, input().split()))
result = compute_powers(n, k, nums)
for value in result:
    print(value)


