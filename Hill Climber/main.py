# run_hill_climber.py

from hc_alg import HillClimber
import pandas as pd

def main():
    """
    Main function to run the HillClimber optimization and handle results.
    """
    # Define file paths
    cell_info_file = '小区信息_filtered.xlsx'
    conflict_inference_file = '冲突及干扰矩阵_filtered.xlsx'
    confusion_file = '混淆矩阵_filtered.xlsx'
    output_file = 'optimized_pci_assignment.xlsx'

    # Define optimization parameters
    num_pcis = 1008
    num_cells = 2890
    max_iterations = 10000

    # Initialize HillClimber
    hill_climber = HillClimber(
        cell_info_file=cell_info_file,
        conflict_inference_file=conflict_inference_file,
        confusion_file=confusion_file,
        num_pcis=num_pcis,
        num_cells=num_cells,
        max_iterations=max_iterations
    )

    # Run hill climbing optimization
    best_assignment, best_MR_sum, best_collision, best_confusion, best_interference = hill_climber.run()

    # Load the original cell information
    df1 = pd.read_excel(cell_info_file)

    # Add the optimized PCI assignments to the DataFrame
    df1['Optimized_PCI'] = best_assignment

    # Save the optimized assignments to a new Excel file
    df1.to_excel(output_file, index=False)

    # Print the results
    print(f"Optimized PCI assignment saved to '{output_file}'.")
    print(f"Best MR_sum: {best_MR_sum}")
    print(f"Collision: {best_collision}")
    print(f"Confusion: {best_confusion}")
    print(f"Interference: {best_interference}")

if __name__ == "__main__":
    main()
