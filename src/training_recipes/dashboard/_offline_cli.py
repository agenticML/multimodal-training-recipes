"""Subprocess entry that always uses the offline CLI."""

from training_recipes.cli import main_offline


def main() -> None:
    main_offline()


if __name__ == "__main__":
    main()
