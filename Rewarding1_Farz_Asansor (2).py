"""
سوال اول امتیازی _ آزمون دو نمونه ای 
"""
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

df = pd.read_parquet(r"C:\Users\Asus\Downloads\divar_clean.parquet")
df = df[["has_elevator", "is_rebuilt"]].dropna()

# گروه بازسازی
rebuilt = df[df["is_rebuilt"] == True]
not_rebuilt = df[df["is_rebuilt"] == False]

# تعداد آسانسوردارها
elevator_rebuilt = (rebuilt["has_elevator"] == True).sum()
elevator_not_rebuilt = (not_rebuilt["has_elevator"] == True).sum()

n_rebuilt = len(rebuilt)
n_not_rebuilt = len(not_rebuilt)

p_rebuilt = elevator_rebuilt / n_rebuilt
p_not_rebuilt = elevator_not_rebuilt / n_not_rebuilt

print("Rebuilt")
print("n =", n_rebuilt)
print("Elevator =", elevator_rebuilt)
print("Proportion =", round(p_rebuilt, 3))

print("\nNot Rebuilt")
print("n =", n_not_rebuilt)
print("Elevator =", elevator_not_rebuilt)
print("Proportion =", round(p_not_rebuilt, 3))

# آزمون دو نمونه ای
count = [elevator_rebuilt, elevator_not_rebuilt]
nobs = [n_rebuilt, n_not_rebuilt]

stat, p_value = proportions_ztest(count, nobs)

print("\nTwo Sample Test")
print("H0: p_rebuilt = p_not_rebuilt")
print("H1: p_rebuilt != p_not_rebuilt")
print("p-value =", p_value)

if p_value < 0.05:
    print("Reject H0")
    print("There is a significant difference.")
else:
    print("Fail to reject H0")
    print("There is no significant difference.")

# نمودار
groups = ["Rebuilt", "Not Rebuilt"]
values = [p_rebuilt * 100, p_not_rebuilt * 100]

plt.bar(groups, values)
plt.ylabel("Elevator proportion (%)")
plt.title("Elevator Proportion by Rebuilt Status")
plt.ylim(0, max(values) * 1.2)

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.show()