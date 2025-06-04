import matplotlib.pyplot as plt
import numpy as np

# Data
fps_allocated_pct = [18.0, 27.0, 36.0, 45.0, 53.9, 71.9, 89.9, 107.9, 125.9, 143.9]
fps_used_pct = [18.0, 27.0, 35.5, 40.8, 48.4, 47.1, 53.0, 55.9, 56.2, 55.7]
fps_values = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot the actual data
plt.plot(fps_allocated_pct, fps_used_pct, 'o-', linewidth=2, markersize=8, 
         color='#2E86AB', label='Actual Usage')

# Add perfect utilization line (45-degree line)
max_val = max(max(fps_allocated_pct), max(fps_used_pct))
plt.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.7, 
         label='Perfect Utilization (100%)')

# Annotate data points with FPS values
for i, (x, y, fps) in enumerate(zip(fps_allocated_pct, fps_used_pct, fps_values)):
    plt.annotate(f'FPS {fps}', (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

# Highlight the peak usage point
peak_idx = fps_used_pct.index(max(fps_used_pct))
plt.scatter(fps_allocated_pct[peak_idx], fps_used_pct[peak_idx], 
           s=150, color='red', zorder=5, alpha=0.7)
plt.annotate(f'Peak Usage\n{fps_used_pct[peak_idx]:.1f}% of baseline', 
             (fps_allocated_pct[peak_idx], fps_used_pct[peak_idx]),
             xytext=(10, -40), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
             fontsize=10, color='red', weight='bold')

# Customize the plot
plt.xlabel('Allocated Subsidies (% of Baseline System Costs)', fontsize=14)
plt.ylabel('Actually Used Subsidies (% of Baseline System Costs)', fontsize=14)
plt.title('Allocated vs. Actually Used Subsidies\n(Mode Share Equity Optimization)', fontsize=16, weight='bold')

# Add grid
plt.grid(True, alpha=0.3)

# Set axis limits with some padding
plt.xlim(0, max(fps_allocated_pct) * 1.05)
plt.ylim(0, max(fps_used_pct) * 1.1)

# Add legend
plt.legend(loc='upper left', fontsize=12)

# Add text box with key insights
textstr = 'Key Insights:\n• Full utilization up to 27% allocation\n• Peak usage at ~56% of baseline costs\n• Diminishing returns beyond 54% allocation'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.show()