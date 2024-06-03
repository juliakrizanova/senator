import pandas as pd
import matplotlib.pyplot as plt


def plot_experiment_results(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    grouped = df.groupby("owner_loss")
    other_loss_min = round(df["other_loss"].min(), 2)
    other_loss_max = round(df["other_loss"].max(), 2)

    for owner_loss, group in grouped:
        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot utilities over iterations
        plt.plot(group["other_loss"], group["u1"], label=f"Senator")
        plt.plot(group["other_loss"], group["u2"], label=f"ANO guy")
        plt.plot(group["other_loss"], group["u3"], label=f"Player 3")

        # Adding labels and title
        plt.xlabel("OTHER LOSS")
        plt.ylabel("Utility")
        plt.title(
            f"Utility of Players for OWNER_LOSS={round(owner_loss,2)} and OTHER_LOSS from {other_loss_min} to {other_loss_max}"
        )
        plt.legend()

        # Display the plot
        filename = f"plots/plot_experiment_results/plot_owner_loss_{round(owner_loss,2)}_other_loss_{other_loss_min}_{other_loss_max}.png"
        plt.savefig(filename)
        plt.close()


# Assuming the CSV file is named 'experiment_results.csv'
# plot_experiment_results("experiment_results.csv")
