import pandas as pd

def update_dataset():
    # Charger le dataset d'origine
    df = pd.read_csv("data/train.csv")

    # Supprimer une colonne peu utile, par exemple "Street"
    # ⚠️ Tu peux changer le nom de la colonne si nécessaire
    if "Street" in df.columns:
        df = df.drop(columns=["Street"])
        print("Colonne 'Street' supprimée du dataset.")
    else:
        print("La colonne 'Street' n'existe pas dans le dataset. Aucune suppression effectuée.")

    # Sauvegarder par-dessus le fichier existant
    df.to_csv("data/train.csv", index=False)
    print("Dataset updated and saved to data/train.csv")

if __name__ == "__main__":
    update_dataset()
