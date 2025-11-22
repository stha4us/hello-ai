from typing import Optional

import click
from timeseries.ingest import main as ingest_features_main
from timeseries.forecast import main as forecast_main
from timeseries.report import main as report_main
from timeseries.train import main as train_main
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
def query(ctx):
    """
    CLI group command for refresh_features.py
    script
    """

    # ====== execute script's main method ======
    run_data_pull_main(
        brand_code_param=ctx.obj["brand_code"],
        output_dir=ctx.obj["output_dir"],
    )

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

@cli.command()
@click.pass_context
def train_model(ctx):
    """
    CLI group command for refresh_features.py
    script
    """

    # ====== execute script's main method ======
    train_main(
        multiprocessing=ctx.obj["multiprocessing"],
    )

@cli.command()
@click.pass_context
def forecast(ctx):
    """
    CLI group command for refresh_features.py
    script
    """

    # ====== execute script's main method ======
    forecast_main(
        multiprocessing=ctx.obj["multiprocessing"],
    )

@cli.command()
@click.pass_context
def report(ctx):
    """
    CLI group command for refresh_features.py
    script
    """

    # ====== execute script's main method ======
    report_main(
        multiprocessing=ctx.obj["multiprocessing"],
    )

if __name__ == "__main__":
    cli()