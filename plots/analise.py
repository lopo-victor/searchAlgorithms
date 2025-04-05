
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("resultados.csv")
sns.set_theme(style="whitegrid")

# Gráfico de Barras Agrupadas: Score por Seed para cada Player
plt.figure(figsize=(14, 7))
sns.barplot(data=df, x="seed", y="score", hue="player", palette="Set2", errorbar=None)
plt.title("Pontuação Player por Seed")
plt.xlabel("Seed")
plt.ylabel("Pontuação")
plt.xticks(rotation=45)
plt.legend(title="Player", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("barras_score_por_seed.png")
plt.show()

# ========== Boxplot: Score por Player ==========
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="player", y="score", hue="player", palette="pastel")
plt.title("Boxplot - Pontuação por Player")
plt.ylabel("Pontuação")
plt.xlabel("Player")
plt.tight_layout()
plt.savefig("boxplot_score_player.png")
plt.show()

# ========== Boxplot: Tempo por Player ==========
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="player", y="execution_time", hue="player", palette="pastel")
plt.title("Boxplot - Tempo de Execução por Player")
plt.ylabel("Tempo de Execução (s)")
plt.xlabel("Player")
plt.tight_layout()
plt.savefig("boxplot_time_player.png")
plt.show()



# ========== Dispersão: Score por Steps ==========
sns.lmplot(
    data=df,
    x="steps",
    y="score",
    hue="player",
    markers="o",
    palette="Set1",
    height=6,
    aspect=1.5,
    ci=None,
    scatter_kws={"s": 60, "alpha": 0.6},
    line_kws={"linewidth": 2}
)
plt.title("Dispersão - Score vs Steps com Linha de Tendência por Player")
plt.xlabel("Passos")
plt.ylabel("Pontuação")
plt.tight_layout()
plt.savefig("score_steps_linha_tendencia.png")
plt.show()


# ========== Correlação adicional observada: Steps vs Time ==========
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="steps", y="execution_time", hue="player")
plt.title("Dispersão - Passos vs Tempo de Execução")
plt.xlabel("Passos")
plt.ylabel("Tempo (s)")
plt.tight_layout()
plt.savefig("dispersao_steps_tempo.png")
plt.show()


