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




