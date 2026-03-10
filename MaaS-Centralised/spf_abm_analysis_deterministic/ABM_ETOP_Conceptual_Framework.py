import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path
from matplotlib.patheffects import withStroke

# Set figure size and DPI for high-quality output
plt.figure(figsize=(10, 14), dpi=300)

# Create a white background
ax = plt.gca()
ax.set_facecolor('white')
plt.axis('off')

# Define professional color palette (colorblind-friendly)
agent_color = '#0072B2'      # Blue
agent_fill = '#E1F0FF'       # Light blue fill
policy_color = '#009E73'     # Green
policy_fill = '#E1FFEF'      # Light green fill
outcome_color = '#D55E00'    # Orange/red
outcome_fill = '#FFEFE1'     # Light orange fill
optimization_color = '#5F509B'  # Purple
optimization_fill = '#EEEBFF'   # Light purple fill
fps_color = '#CC9900'        # Gold
fps_fill = '#FFFAE6'         # Light gold fill

# Give more vertical space between main sections - ADJUSTED FOR BETTER SPACING
agent_y = 0.02
policy_y = 0.25
outcome_y = 0.55  # Slightly adjusted
optimization_y = 0.8  # Slightly adjusted

# Function to draw a box with title outside the box
def draw_section(y_pos, width, height, title, color, fill_color, fontsize=14):
    # Draw shadow
    shadow = patches.FancyBboxPatch((0.5-width/2+0.01, y_pos-0.01), width, height, 
                             boxstyle=patches.BoxStyle.Round(pad=0.03),
                             linewidth=0, facecolor='gray', alpha=0.2,
                             zorder=0, clip_on=False)
    ax.add_patch(shadow)
    
    # Draw main box
    rect = patches.FancyBboxPatch((0.5-width/2, y_pos), width, height, 
                            boxstyle=patches.BoxStyle.Round(pad=0.03),
                            linewidth=2.5, edgecolor=color, facecolor=fill_color, alpha=0.9,
                            zorder=1, clip_on=False)
    ax.add_patch(rect)
    
    # Add title with subtle text effect (positioned above the box)
    plt.text(0.5, y_pos+height+0.025, title, ha='center', va='bottom', 
             fontsize=fontsize, weight='bold', color='black',
             path_effects=[withStroke(linewidth=3, foreground='white')])
    return rect

# Function to draw a rounded box with improved styling
def draw_box(x, y, width, height, title, color, fill_color, fontsize=11, bold=False):
    # Draw shadow
    shadow = patches.FancyBboxPatch((x-width/2+0.005, y-height/2-0.005), width, height, 
                             boxstyle=patches.BoxStyle.Round(pad=0.02),
                             linewidth=0, facecolor='gray', alpha=0.2,
                             zorder=1, clip_on=False)
    ax.add_patch(shadow)
    
    # Draw main box
    rect = patches.FancyBboxPatch((x-width/2, y-height/2), width, height, 
                            boxstyle=patches.BoxStyle.Round(pad=0.02),
                            linewidth=2, edgecolor=color, facecolor=fill_color,
                            zorder=2, clip_on=False)
    ax.add_patch(rect)
    
    # Add title with better formatting
    weight = 'bold' if bold else 'normal'
    plt.text(x, y, title, ha='center', va='center', fontsize=fontsize, 
             color='black', weight=weight,
             path_effects=[withStroke(linewidth=2, foreground='white')])
    return rect

# Function to draw an arrow with improved styling
def draw_arrow(start_x, start_y, end_x, end_y, color, style='-', width=2, zorder=1, label=None, rad=0.1):
    dx = end_x - start_x
    dy = end_y - start_y
    arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                    connectionstyle=f"arc3,rad={rad}", 
                                    arrowstyle="-|>", color=color, linewidth=width,
                                    linestyle=style, zorder=zorder)
    ax.add_patch(arrow)
    
    if label:
        # Calculate position for label along the arrow
        label_x = start_x + dx/2
        label_y = start_y + dy/2
        
        # Adjust for arc/curved arrows
        if rad != 0:
            # Create a more appropriate offset based on the arc
            if abs(dx) > abs(dy):  # More horizontal arrow
                offset_x = 0
                offset_y = 0.02 if rad > 0 else -0.02
            else:  # More vertical arrow
                offset_x = 0.02 if rad > 0 else -0.02
                offset_y = 0
            
            # Apply the offset
            label_x += offset_x
            label_y += offset_y
        
        # Create a background for the label
        plt.text(label_x, label_y, label, 
                ha='center', va='center', fontsize=9, weight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, 
                         boxstyle='round,pad=0.2', linewidth=1),
                path_effects=[withStroke(linewidth=2, foreground='white')],
                zorder=zorder+1)

# Add main title at top of the diagram - IMPROVED POSITIONING
plt.text(0.5, 0.99, "ABM-ETOP Conceptual Framework", 
         ha='center', va='top', fontsize=18, weight='bold',
         path_effects=[withStroke(linewidth=4, foreground='white')])

# Draw the main sections with improved spacing
agent_section = draw_section(agent_y, 0.92, 0.145, "Agent Interactions Layer", agent_color, agent_fill, 14)
policy_section = draw_section(policy_y, 0.92, 0.21, "Policy Intervention Layer", policy_color, policy_fill, 14)
outcome_section = draw_section(outcome_y, 0.92, 0.171, "System-Level Outcomes Layer", outcome_color, outcome_fill, 14)
optimization_section = draw_section(optimization_y-0.001, 0.92, 0.12, "Optimization Process", optimization_color, optimization_fill, 14)

# Agent Layer Components - FIXED POSITIONING AND TEXT
commuter_x, commuter_y = 0.25, agent_y + 0.085
coordinator_x, coordinator_y = 0.5, agent_y + 0.085
provider_x, provider_y = 0.75, agent_y + 0.085

# FIXED: Separate text elements for clearer labels
draw_box(commuter_x, commuter_y, 0.16, 0.1, "", agent_color, agent_fill)  # Empty box first
plt.text(commuter_x, commuter_y+0.01, "Commuter \nAgents", ha='center', va='center', fontsize=11, weight='bold')
plt.text(commuter_x, commuter_y-0.02, "(socioeconomic \ndiversity)", ha='center', va='center', fontsize=10)

draw_box(coordinator_x, coordinator_y, 0.16, 0.1, "", optimization_color, optimization_fill)  # Empty box first
plt.text(coordinator_x, coordinator_y+0.01, "System \nCoordinator", ha='center', va='center', fontsize=11, weight='bold')
plt.text(coordinator_x, coordinator_y-0.02, "(MaaS Platform)", ha='center', va='center', fontsize=10)

draw_box(provider_x, provider_y, 0.16, 0.1, "", policy_color, policy_fill)  # Empty box first
plt.text(provider_x, provider_y+0.01, "Service Provider \nAgents", ha='center', va='center', fontsize=11, weight='bold')
plt.text(provider_x, provider_y-0.02, "(transport modes)", ha='center', va='center', fontsize=10)

travel_request_arrow = draw_arrow(commuter_x+0.125, commuter_y, coordinator_x-0.125, coordinator_y, agent_color, rad=0.1)
plt.text((commuter_x+0.125 + coordinator_x-0.125)/2, commuter_y+0.05, "Travel\nrequests", 
         ha='center', va='center', fontsize=8, weight='bold',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor=agent_color, 
                  boxstyle='round,pad=0.3', linewidth=1),
         zorder=10)

# From Coordinator to Commuter
travel_options_arrow = draw_arrow(coordinator_x-0.125, coordinator_y-0.04, commuter_x+0.125, commuter_y-0.04, agent_color, rad=-0.1)
plt.text((coordinator_x-0.125 + commuter_x+0.125)/2, commuter_y-0.065, "Travel\noptions", 
         ha='center', va='center', fontsize=8, weight='bold',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor=agent_color, 
                  boxstyle='round,pad=0.3', linewidth=1),
         zorder=10)

# From Coordinator to Provider
demand_patterns_arrow = draw_arrow(coordinator_x+0.125, coordinator_y, provider_x-0.125, provider_y, policy_color, rad=0.1)
plt.text((coordinator_x+0.125 + provider_x-0.125)/2, provider_y+0.05, "Demand\npatterns", 
         ha='center', va='center', fontsize=8, weight='bold',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor=policy_color, 
                  boxstyle='round,pad=0.3', linewidth=1),
         zorder=10)

# From Provider to Coordinator
service_avail_arrow = draw_arrow(provider_x-0.125, provider_y-0.04, coordinator_x+0.125, coordinator_y-0.04, policy_color, rad=-0.1)
plt.text((provider_x-0.125 + coordinator_x+0.125)/2, coordinator_y-0.065, "Service\navailability", 
         ha='center', va='center', fontsize=8, weight='bold',
         bbox=dict(facecolor='white', alpha=0.85, edgecolor=policy_color, 
                  boxstyle='round,pad=0.3', linewidth=1),
         zorder=10)

# Policy Layer Components
fps_x, fps_y = 0.25, policy_y + 0.085
# Moved matrix to give it more space
matrix_x, matrix_y = 0.68, policy_y + 0.085

draw_box(fps_x, fps_y, 0.25, 0.11, "Fixed Pool Subsidy (FPS)", fps_color, fps_fill, 11, True)

# IMPROVED: Policy Intervention Layer visualization - CLEANER VERSION
# Create a container for the allocation visualization
alloc_box = draw_box(matrix_x, matrix_y, 0.45, 0.13, "", policy_color, policy_fill)

# Add title
plt.text(matrix_x, matrix_y+0.055, "Subsidy Allocation Framework", ha='center', va='center', 
         fontsize=11, weight='bold')

# Create a 3x4 grid for allocation visualization
income_groups = ['Low', 'Middle', 'High']
transport_modes = ['Bike', 'Car', 'Public', 'MaaS']

# Define grid positions
grid_width, grid_height = 0.35, 0.07
grid_left = matrix_x - 0.15  # Centered position
grid_bottom = matrix_y - 0.045

# Calculate cell dimensions
cell_width = grid_width / len(transport_modes)
cell_height = grid_height / len(income_groups)

# Draw income labels on left side
for i, income in enumerate(income_groups):
    y_pos = grid_bottom + (i + 0.5) * cell_height
    plt.text(grid_left - 0.02, y_pos, income, ha='right', va='center', 
             fontsize=8, weight='bold', color=policy_color)

# Draw mode labels on top
for j, mode in enumerate(transport_modes):
    x_pos = grid_left + (j + 0.5) * cell_width
    plt.text(x_pos, grid_bottom + grid_height + 0.01, mode, ha='center', va='bottom', 
             fontsize=8, weight='bold', color=policy_color)

# Define allocation values - higher values for low income
allocations = {
    'Low': {'Bike': 0.65, 'Car': 0.5, 'Public': 0.7, 'MaaS': 0.55},
    'Middle': {'Bike': 0.4, 'Car': 0.45, 'Public': 0.35, 'MaaS': 0.3},
    'High': {'Bike': 0.15, 'Car': 0.05, 'Public': 0.2, 'MaaS': 0.15}
}

# Draw the grid and allocation circles
for i, income in enumerate(income_groups):
    for j, mode in enumerate(transport_modes):
        x = grid_left + j * cell_width
        y = grid_bottom + i * cell_height
        
        # Draw cell outline
        rect = patches.Rectangle((x, y), cell_width, cell_height, 
                                linewidth=1, edgecolor=policy_color,
                                facecolor='white', alpha=0.1,
                                zorder=3)
        ax.add_patch(rect)
        
        # Add circle with size based on allocation value
        circle_size = allocations[income][mode] * 0.015  # Scale factor for circle size
        circle = plt.Circle((x + cell_width/2, y + cell_height/2), 
                           circle_size, color=policy_color, 
                           alpha=0.9, zorder=4)
        ax.add_patch(circle)

# Add legend at bottom of allocation grid
plt.text(matrix_x, grid_bottom - 0.015, 
        "Circle size & cell color = allocation percentage", 
        ha='center', va='center', fontsize=8, style='italic')
# FPS to matrix connection
draw_arrow(fps_x+0.125, fps_y, matrix_x-0.21, matrix_y, fps_color, label="Distributes", rad=0.1)

# Outcome Layer Components
mode_equity_x, mode_equity_y = 0.25, outcome_y + 0.045
time_equity_x, time_equity_y = 0.5, outcome_y + 0.045
system_time_x, system_time_y = 0.75, outcome_y + 0.045
tradeoffs_x, tradeoffs_y = 0.5, outcome_y + 0.125

# Draw outcome components
draw_box(mode_equity_x, mode_equity_y+0.01, 0.2, 0.035, "Mode Share Equity\n(vertical equity 1)", outcome_color, outcome_fill, 11, True)
draw_box(time_equity_x, time_equity_y+0.01, 0.2, 0.035, "Travel Time Equity\n(vertical equity 2)", outcome_color, outcome_fill, 11, True)
draw_box(system_time_x, system_time_y+0.01, 0.2, 0.045, "Total System \nTravel Time\n(horizontal equity \n & efficiency)", policy_color, policy_fill, 11, True)
draw_box(tradeoffs_x, tradeoffs_y+0.025, 0.5, 0.02, "Equity-Efficiency Trade-offs", outcome_color, outcome_fill, 11, True)

# Connect outcomes to tradeoffs with improved arrows
draw_arrow(mode_equity_x, mode_equity_y, tradeoffs_x-0.15, tradeoffs_y, outcome_color, rad=0.05)
draw_arrow(time_equity_x, time_equity_y+0.045, tradeoffs_x, tradeoffs_y, outcome_color, rad=0.1)
draw_arrow(system_time_x, system_time_y+0.045, tradeoffs_x+0.15, tradeoffs_y, policy_color, rad=0.0)

# Optimization Process Components - spread out more
sim_x, sim_y = 0.2, optimization_y + 0.085
eval_x, eval_y = 0.4, optimization_y + 0.085
opt_x, opt_y = 0.6, optimization_y + 0.085
update_x, update_y = 0.8, optimization_y + 0.085

# Draw optimization components
draw_box(sim_x, sim_y, 0.15, 0.05, "Simulation\nRun ABM", optimization_color, optimization_fill, 11, True)
draw_box(eval_x, eval_y, 0.15, 0.05, "Evaluation\nMetrics", optimization_color, optimization_fill, 11, True)
draw_box(opt_x, opt_y, 0.15, 0.05, "Bayesian\nOptimization", optimization_color, optimization_fill, 11, True)
draw_box(update_x, update_y, 0.15, 0.05, "Policy\nUpdate", optimization_color, optimization_fill, 11, True)

# Connect optimization components with a cycle of arrows
draw_arrow(sim_x+0.075, sim_y, eval_x-0.075, eval_y, optimization_color, width=1.8, rad=0.0)
draw_arrow(eval_x+0.075, eval_y, opt_x-0.075, opt_y, optimization_color, width=1.8, rad=0.0)
draw_arrow(opt_x+0.075, opt_y, update_x-0.075, update_y, optimization_color, width=1.8, rad=0.0)
# Complete the cycle with a return arrow
draw_arrow(update_x-0.035, update_y-0.05, sim_x+0.025, sim_y-0.05, optimization_color, width=1.8, rad=-0.2, label="Iterate")

# Connect across layers with improved dashed arrows
# Policy to Outcomes
draw_arrow(matrix_x-0.05, matrix_y-0.005, mode_equity_x, mode_equity_y, outcome_color, style='--', width=2.0, rad=-0.2, label="Impacts")
draw_arrow(matrix_x, matrix_y-0.02, time_equity_x, time_equity_y, outcome_color, style='--', width=2.0, rad=-0.1, label="Affects")
draw_arrow(matrix_x+0.18, matrix_y-0.02, system_time_x, system_time_y, policy_color, style='--', width=2.0, rad=-0.2, label="Influences")


# Make the policy update arrow more distinct
opt_policy_arrow = draw_arrow(update_x+0.078, update_y+0.048, matrix_x+0.23, matrix_y+0.065, optimization_color, style='--', width=2, rad=-0.17, label="Update \nallocation")

# Set axes limits with padding for better spacing
plt.xlim(0, 1)
plt.ylim(0, 1)

feedback_path = Path([
    (tradeoffs_x+0.25, tradeoffs_y), 
    (eval_x, eval_y-0.05),
    (update_x, update_y-0.05), 
    (matrix_x+0.2, matrix_y+0.065),
    (matrix_x-0.18, matrix_y+0.065),
    (mode_equity_x, mode_equity_y-0.045),
    (mode_equity_x, mode_equity_y+0.045),
    (tradeoffs_x-0.15, tradeoffs_y-0.03),
    (tradeoffs_x+0.25, tradeoffs_y)
])
feedback_patch = patches.PathPatch(feedback_path, facecolor='none', edgecolor=optimization_color, 
                                  alpha=0.2, linewidth=2, linestyle='-', zorder=0)
ax.add_patch(feedback_patch)

plt.savefig('abm_etop_framework_fixed.png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.show()