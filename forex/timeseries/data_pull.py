from pathlib import Path
from src.utils import SnowflakeContext
from src.utils import PROJECT_ROOT

def main(
    brand_code_param=None,
    output_dir=None,
):
    """
    TODO: Add docstring
    """
    output_path = Path(PROJECT_ROOT) / "data/raw" / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    # ====== fetch query files ======
    with open(
        Path(PROJECT_ROOT) / "sql/data.sql", "r"
    ) as infile:
        query = infile.read()

    # ====== add conditions to query ======
    if isinstance(brand_code_param, str):
        predicates = [f"'{brand_code_param}'"]
    else:
        predicates = [f"'{param}'" for param in brand_code_param]
    clause = "{}".format(",".join(predicates))

    with SnowflakeContext() as ctx:
        # === initialize connection cursor ===
        cursor = ctx._conn.cursor()

        # === set cursor variables ===
        cursor.execute(f"SET BRAND_CODE_PARAM = {clause};")

        # === execute query and fetch result ===
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        df.to_csv(Path(output_path) / "data.csv")


if __name__=='__main__':
    main()