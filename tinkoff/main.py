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


# Задача 4
n, x, y, z = map(int, input().split())
a = list(map(int, input().split()))

min_ops = float('inf')
best_indices = []
best_increments = []

for i in range(n):
    for j in range(n):
        for k in range(n):
            ops_x = (x - (a[i] % x)) % x
            ops_y = (y - (a[j] % y)) % y
            ops_z = (z - (a[k] % z)) % z

            if i == j:
                ops_y = max(0, ops_y - ops_x)
            if i == k:
                ops_z = max(0, ops_z - ops_x)
            if j == k:
                ops_z = max(0, ops_z - ops_y)

            total_ops = ops_x + ops_y + ops_z

            if total_ops < min_ops:
                min_ops = total_ops
                best_indices = [i, j, k]
                best_increments = [ops_x, ops_y, ops_z]

print(min_ops)


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


# Задача 6
def check_collinearity(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def find_max_non_collinear_triplets(points):
    n = len(points)
    triplets = []
    used = [False] * n

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if not check_collinearity(points[i], points[j], points[k]):
                    triplets.append((i, j, k))

    triplets.sort()

    max_count = 0

    def backtrack(index, count):
        nonlocal max_count
        if index == len(triplets):
            max_count = max(max_count, count)
            return
        backtrack(index + 1, count) 

        i, j, k = triplets[index]
        if not (used[i] or used[j] or used[k]):
            used[i] = used[j] = used[k] = True
            backtrack(index + 1, count + 1)
            used[i] = used[j] = used[k] = False

    backtrack(0, 0)
    return max_count

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]
print(find_max_non_collinear_triplets(points))


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
