import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from senator.game import *
from senator.utility import *

# V souboru utility.py zmen parameter PLAYERS, aby odpovidal poctu kandidatu

# Do tohoto souboru bude třeba doplnit ownership hráčů (vlastní oblast = 1, má částečnou většinu = 0.5) a
# k tomu příslušné utility (vlastní oblast = 0.5* počet hlasů oblasti, má částečnou většinu = 0.25* počet hlasů oblasti)
FILE_PATH = "w0_data.csv"

# Tento parametr bude chtít před každým spuštěním změnit, jinak se přepíšou data v souborech!
WEEK = 2

# Nastavit podle toho, kolik návštěv za dýne bude chtít stihnout. Každý týden to může být jinak
LAST_WEEK_LENGTH = 7  # Počet dní v posledním období
WEEK_LENGTH = 7  # Počet dní v aktuálním období

# Doplnit poporade oblasti, jak je senátor navštívil v posledním období
# Pokud se senator presne ridil planem, nastavit na None
this_week_actions = None

# Hodnoty OWNER_LOSS a OTHER_LOSS jsou nastavene dost od oka. V souboru parameters.pdf je uvedeno, kolik procent hlasu zbyde za 5/6/7 dnu, pokud mame dane hodnoty parametru
OWNER_LOSS = 0.9
OTHER_LOSS = 0.85

""" DOPOCITANI DAT Z NAVSTEV AKTUALNIHO TYDNE TYDNE (mimo prvni tyden)
    1. Nacist data z predchoziho tydne
    2. Inicializovat hru
    3. Spustit greedy search pro spočtení akcí ostatních hráčů
    4. Spočítat modifikovanou utilitu (po zadání akcí senátora)
    5. Uložit data do souboru
"""
if WEEK != 0:
    # Načítání dat z předchozího týdne, inicializace hry a spuštění greedy search pro spočtení akcí ostatních hráčů
    votes, initial_utility, is_owner = parse_data(load_data(f"w{WEEK - 1}_data.csv"))
    game = Game(initial_utility, votes, is_owner, OWNER_LOSS, OTHER_LOSS)
    actions_in_steps, utility, strategies_in_steps = game.run_greedy_search(
        LAST_WEEK_LENGTH
    )

    # (Pokud by senatáro dodával data i za jiné kandidáty, tak by se to muselo upravit zde)
    # Jestli nam senator rekl ze se minuly tyden choval jinak nez podle planu, tady rikame to co senator skutecne delal
    if this_week_actions is not None:
        for i in range(len(actions_in_steps)):
            actions_in_steps[i][0] = this_week_actions[i]

    modified_utility = game.compute_modified_utility(actions_in_steps, votes)

    # Transpose the array to match the CSV structure
    modified_utility = modified_utility.T

    df = pd.read_csv(FILE_PATH, delimiter=";")

    # Update the columns u1, u2, u3 with new values
    df[["u1", "u2", "u3"]] = modified_utility

    # Save the updated DataFrame to a new CSV file
    output_file_path = f"w{WEEK}_data.csv"
    df.to_csv(output_file_path, sep=";", index=False)

""" VYTVORENI GRAFU PRO SENATORA
    1. Načtení dat z CSV souboru
    2. Inicializace hry
    3. Spuštění greedy search pro spočtení akcí
    4. Spočítání počtu návštěv v jednotlivých oblastech senátorem
    5. Vytvoření grafu
"""
votes, initial_utility, is_owner = parse_data(load_data(f"w{WEEK}_data.csv"))
game_next = Game(initial_utility, votes, is_owner, OWNER_LOSS, OTHER_LOSS)
actions_in_steps, utility, strategies_in_steps = game_next.run_greedy_search(
    WEEK_LENGTH
)

senator_actions = np.array([arr[0] for arr in actions_in_steps])
weekly_actions = np.bincount(senator_actions, minlength=17)


plt.figure(figsize=(14, 8))

# Čísla oblastí 1 až 17
values = np.arange(len(weekly_actions)) + 1

# PLOTĚNÍ GRAFU
plt.bar(
    values,
    weekly_actions,
    width=0.5,
    label=f"Období: {WEEK} (počet dní: {WEEK_LENGTH})",
)

plt.title(f"Návštěvy na následující období")
plt.xlabel("Oblast")
plt.ylabel("Počet návštěv")
plt.xticks(values)
plt.legend()
plt.grid(True)

# Save the plot
filename = f"{WEEK}_tyden_plan_akci.png"
plt.savefig(filename)
plt.close()
