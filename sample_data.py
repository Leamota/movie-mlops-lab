import matplotlib.pyplot as plt

# Example genre counts before and after drift
genres = ["Action", "Comedy", "Drama", "Documentary"]
before = [50, 30, 20, 10]
after = [20, 25, 40, 30]

x = range(len(genres))

plt.bar(x, before, width=0.4, label="Before Drift", align="center")
plt.bar([i + 0.4 for i in x], after, width=0.4, label="After Drift", align="center")

plt.xticks([i + 0.2 for i in x], genres)
plt.ylabel("Number of Movies")
plt.title("Genre Distribution Before vs After Drift")
plt.legend()
plt.tight_layout()
plt.savefig("images/genre_distribution.png")
plt.show()
