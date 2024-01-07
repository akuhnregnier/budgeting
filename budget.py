from pathlib import Path
import numpy as np
import re
import pandera as pa
import functools
from pprint import pprint
import io
import pydantic
from typing import TypeAlias, assert_never
import itertools
import yaml
import json
import pandas as pd
from typing import Final
from loguru import logger


class BudgetingException(Exception):
    ...


class InternalError(BudgetingException):
    """Something internal has gone wrong."""


class ConfigError(BudgetingException):
    """Configuration is invalid."""


class TransactionError(BudgetingException):
    """Invalid/unexpected transactions."""


CURR_DIR: Final = Path(__file__).resolve().parent
CONFIG_DIR: Final = CURR_DIR / "config"


class AccountInfo(pydantic.BaseModel):
    name: str
    balance: str
    available_balance: str


def truncate(s: str, /, n: int) -> str:
    """Truncate `s` to at most `n` characters."""
    if len(s) <= n:
        return s
    sub = "..."
    s = s[: (n - len(sub))]
    return s + sub


def get_category_file() -> Path | None:
    if not CONFIG_DIR.is_dir():
        raise ConfigError(f"{CONFIG_DIR} is not a directory.")
    category_files = tuple(
        itertools.islice(
            itertools.chain(
                CONFIG_DIR.glob("*.yml"),
                CONFIG_DIR.glob("*.yaml"),
            ),
            0,
            2,
        )
    )

    match len(category_files):
        case 2:
            raise ConfigError("Multiple category configs found.")
        case 1:
            return category_files[0]
        case 0:
            return None
        case _:
            raise InternalError("Unreachable.")

    assert_never(0)


RawCategoryPatterns = pydantic.RootModel[dict[str, tuple[str, ...]]]
CategoryPatterns: TypeAlias = dict[str, tuple[re.Pattern, ...]]


def get_category(s: str, patterns: CategoryPatterns) -> str | None:
    """Get category of `s` based on `patterns`."""
    for category, category_patterns in patterns.items():
        if any(pattern.match(s) for pattern in category_patterns):
            return category
    return None


@pa.check_input(
    pa.DataFrameSchema(
        {
            "Description": pa.Column(str),
            "Paid out": pa.Column(float, nullable=True),
            "Paid in": pa.Column(float, nullable=True),
        }
    )
)
def remove_cancelled_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    If transactions with the same description appear twice, with equal amounts once in
    'Paid out' and then in 'Paid in', remove these transaction.

    """
    data = data.copy()

    indices_to_remove: list[int] = []
    for description, group in data.groupby("Description")[["Paid out", "Paid in"]]:
        if (n := group.shape[0]) == 1:
            continue
        logger.trace(
            "'{description}' matched {n} transactions.", description=description, n=n
        )
        outs: set[float] = set(group["Paid out"].tolist())
        ins: set[float] = set(group["Paid in"].tolist())
        if any(shared := outs.intersection(ins)):
            for amount in shared:
                shared_group = group.loc[
                    (np.isclose(group["Paid out"], amount))
                    | (np.isclose(group["Paid in"], amount))
                ]
                shared_out = shared_group["Paid out"]
                shared_out = shared_out[~pd.isna(shared_out)]
                shared_in = shared_group["Paid in"]
                shared_in = shared_in[~pd.isna(shared_in)]
                total_sum = np.sum(shared_out + shared_in)
                if not np.isclose(total_sum, 0):
                    raise TransactionError(f"{total_sum=}, but expected 0.")
                indices_to_remove.extend(shared_group.index.tolist())
    logger.info(
        "Removing {n_remove} cancelled transactions.", n_remove=len(indices_to_remove)
    )
    data = data.drop(indices_to_remove, axis="index")
    return data


def main():
    file: Final = list(CURR_DIR.glob("data/*.csv"))[0]
    logger.info("File: {file}", file=file)
    raw_text = file.read_text(encoding="latin1")
    account_info: AccountInfo | None = None

    category_config_file: Final = get_category_file()
    category_patterns: CategoryPatterns | None = None
    if category_config_file:
        logger.info(
            f"Loading category config: {category_config_file}",
            category_config_file=category_config_file,
        )
        with category_config_file.open("r") as f:
            category_patterns = {
                key: tuple(
                    map(functools.partial(re.compile, flags=re.I), pattern_strings)
                )
                for key, pattern_strings in RawCategoryPatterns(yaml.safe_load(f))
                .model_dump()
                .items()
            }
        print("Categories:")
        pprint(category_patterns)

    csv_file = file
    if raw_text.startswith('"Account Name:"'):
        # Analyse and strip leading lines.
        lines = raw_text.split("\n")

        name_line_parts = lines.pop(0).split(",")
        assert "Account Name:" in name_line_parts[0]

        balance_line_parts = lines.pop(0).split(",")
        assert "Account Balance:" in balance_line_parts[0]

        available_balance_line_parts = lines.pop(0).split(",")
        assert "Available Balance:" in available_balance_line_parts[0]

        account_info = AccountInfo(
            name=json.loads(name_line_parts[1]),
            balance=json.loads(balance_line_parts[1]),
            available_balance=json.loads(available_balance_line_parts[1]),
        )
        csv_file = io.StringIO("\n".join(filter(None, lines)))

    logger.info("Account info:\n{account_info}", account_info=account_info)
    data = pd.read_csv(csv_file)
    for col in ("Paid in", "Paid out", "Balance"):
        data[col] = (
            data[col]
            .apply(lambda x: x.removeprefix("£") if isinstance(x, str) else x)
            .astype("float")
        )
    print(data)
    print(data.info())
    data = remove_cancelled_transactions(data)

    category_map: dict[str, str | None] = {}
    print(f"# of transactions: {data.shape[0]}")
    print("Descriptions:")
    for i, description in enumerate(data["Description"].unique()):
        category = None
        if category_patterns:
            category = get_category(description, category_patterns) or category
        category_map[description] = category
        print(f"{i:03d} - {truncate(category or '', 20):>20s} - {description}")

    data["Category"] = data["Description"].apply(lambda key: category_map.get(key))
    print(data)

    uncategorised = data["Category"].apply(lambda x: x is None)
    if np.any(uncategorised):
        print("Uncategorised transaction descriptions:")
        for i, (description, group) in enumerate(
            (data.sort_values("Paid in").loc[uncategorised]).groupby("Description")
        ):
            out_sum = -group["Paid out"].sum()
            in_sum = group["Paid in"].sum()
            print(
                f"{i:03d} | {truncate(description, 50):>50s} | {out_sum:>+9.2f} | {in_sum:>+9.2f}"
            )

    for category, group in data.groupby("Category")[
        ["Date", "Description", "Paid out", "Paid in"]
    ]:
        print(f"\nCategory: {category}")
        print(group)

    print(
        data[["Category", "Paid out", "Paid in"]]
        .groupby("Category")
        .sum()
        .sort_values("Paid out", ascending=False)
    )


if __name__ == "__main__":
    main()
