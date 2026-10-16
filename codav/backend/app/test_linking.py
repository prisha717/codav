from app.dataset_files import load_dataset_df
from app.dataset_linking import shared_columns_exact, suggest_links, suggestions_to_dict

def main():
    ds_a_name = "OECD Inflation"
    ds_b_name = "World Bank Inflation (Time Series)"

    df_a = load_dataset_df(ds_a_name)
    df_b = load_dataset_df(ds_b_name)

    print("SharedExact")
    print(shared_columns_exact(df_a, df_b))

    print("SuggestedLinks")
    sugs = suggest_links(df_a, df_b, top_k=10, name_threshold=0.65)
    print(suggestions_to_dict(sugs))

if __name__ == "__main__":
    main()