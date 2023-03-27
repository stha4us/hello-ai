from typing import Optional

import click
from timeseries.ingest import main as ingest_features_main
from timeseries.utils import GroupWithCommandOptions


# ===== Main workflow group =====
@click.group(cls=GroupWithCommandOptions)
@click.option("-o", "--output-dir", help="Output directory name")
@click.option(
    "--multiprocessing/--no-multiprocessing",
    is_flag=True,
    default=False,
    help="Turn multiprocessing on or off",
)
@click.option(
    "--plot/--no-plot",
    is_flag=True,
    default=False,
    help="Flag to generate forecast plots",
)
@click.pass_context
def cli(
    ctx,
    output_dir: Optional[str] = None,
    multiprocessing: Optional[bool] = None,
    plot: Optional[bool] = None,
):
    # ===== make dictionary context object =====
    ctx.obj = dict()
    ctx.obj["output_dir"] = output_dir
    ctx.obj["multiprocessing"] = multiprocessing
    ctx.obj["plot"] = plot

@cli.command()
@click.pass_context
def ingest_features(ctx):
    """
    CLI group command for refresh_features.py
    script
    """

    # ====== execute script's main method ======
    ingest_features_main(
        multiprocessing=ctx.obj["multiprocessing"],
    )


if __name__ == "__main__":
    cli()