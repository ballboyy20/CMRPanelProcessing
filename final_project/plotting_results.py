import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from math import degrees


def main():
	panel_results = r"C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\position_results\all - panel_positions.csv"
	flat_results = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\position_results\flat_scans\all - panel_positions.csv'
	#full_df = pd.read_csv(all_results)
	graph2(panel_results, flat_results)#, save_dir = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\graph_results\flat_all_metrics_comparison.png')


def graph1(panel_data, flat_data, save_dir=None):
	full_df = pd.concat((pd.read_csv(panel_data), pd.read_csv(flat_data)))
	full_df['theta_x'] = [degrees(value) for value in (full_df['theta_x'])]
	full_df['theta_y'] = [degrees(value) for value in (full_df['theta_y'])]
	
	sns.set(font_scale=0.5)
	fig = plt.figure(figsize=(10, 6))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(223)
	ax3 = fig.add_subplot(224)
	t_x = sns.scatterplot(ax=ax1, data=full_df, x='Joint Type', y='theta_x', hue='Joint Type', marker="_", s=500, linewidths=0.75)
	t_x.legend(fontsize=5)
	t_y = sns.scatterplot(ax=ax2, data=full_df, x='Joint Type', y='theta_y', hue='Joint Type', marker="_", s=200, linewidths=0.75)
	t_y.legend_.remove()
	d_z = sns.scatterplot(ax=ax3, data=full_df, x='Joint Type', y='delta_z', hue='Joint Type', marker="_", s=200, linewidths=0.75)
	d_z.legend_.remove()
	
	save_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\graph_results\flat_all_metrics_comparison.png'
	if save_dir:
		plt.savefig(save_file, dpi=300)
	plt.show()


def graph2(panel_data, save_dir=None):
	# TODO: Rename prototypes to P1, P2, etc.
	full_df = pd.read_csv(panel_data)
	full_df['theta_x'] = [degrees(value) for value in (full_df['theta_x'])]
	full_df['theta_y'] = [degrees(value) for value in (full_df['theta_y'])]
	
	sns.set(font_scale=0.5)
	fig = plt.figure(figsize=(10, 6))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(223)
	ax3 = fig.add_subplot(224)
	t_x = sns.scatterplot(ax=ax1, data=full_df, x='Joint Type', y='theta_x', marker="_", s=500,
						  linewidths=0.75)
	ax1.xaxis.grid(False)
	t_y = sns.scatterplot(ax=ax2, data=full_df, x='Joint Type', y='theta_y', marker="_", s=200,
						  linewidths=0.75)
	ax2.xaxis.grid(False)
	d_z = sns.scatterplot(ax=ax3, data=full_df, x='Joint Type', y='delta_z', marker="_", s=200,
						  linewidths=0.75)
	ax3.xaxis.grid(False)
	
	save_file = r'C:\Users\thetk\Documents\BYU\Work\pythonProject\final_project\graph_results\flat_all_metrics_comparison.png'
	if save_dir:
		plt.savefig(save_file, dpi=300)
	plt.show()


if __name__ == '__main__':
	main()
