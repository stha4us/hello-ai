"""Utilities for workflow scripts"""
import click
import os
import snowflake.connector

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())

class DatabaseContext:
    """Database context manager."""

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self._conn:
            self._conn.close()

    def close(self):
        if self._conn:
            self._conn.close()


class SnowflakeContext(DatabaseContext):
    """
    Snowflake context manager. Establishes
    snowflake connection, fetches credentials
    and hadles query execution. Used as a
    context manager for the query objects in
    :mod:`t2.database.strategies` module.

    Attributes
    ----------
    username : str
        Snowflake username.
    password : str
        Snowflake password.
    wareHouse : str, default 'PROD_WH'
        Snowflake warehouse.
    dataBase : str, default 'PROD_DB'
        Snowflake database.
    schema : str, default 'PROD'
        Snowflake schema.
    account : str
        Snowflake account name.
    role : str

    Parameters
    ----------
    username : str
        Snowflake username.
    password : str
        Snowflake password.
    wareHouse : str, default 'PROD_WH'
        Snowflake warehouse.
    dataBase : str, default 'PROD_DB'
        Snowflake database.
    schema : str, default 'PROD'
        Snowflake schema.
    account : str
        Snowflake account name.
    role : str
    """

    def __init__(
        self,
        username=None,
        password=None,
        warehouse="PROD_WH",
        database="PROD_DB",
        schema="PROD",
        account,
        role,
        connection=None,
    ):

        # ====== Specifying query parametrization format ======
        snowflake.connector.paramstyle = "pyformat"

        if connection:
            self._conn = connection

        else:

            # ====== check snowflake environment is setup ======
            if not os.environ.get("SNOWFLAKE_USER"):
                raise ValueError(
                    "Could not read snowflake username from " "environment."
                )

            # ====== Setting connector parameters ======
            username = os.environ.get("SNOWFLAKE_USER")
            password = os.environ.get("SNOWFLAKE_PASS")
            warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
            database = os.environ.get("SNOWFLAKE_DATABASE")
            schema = os.environ.get("SNOWFLAKE_SCHEMA")
            account = os.environ.get("SNOWFLAKE_ACCOUNT")
            role = os.environ.get("SNOWFLAKE_ROLE")
            insecure_mode = True

            # ====== making parameter dictionary ======
            params = dict(
                user=username,
                password=password,
                warehouse=warehouse,
                database=database,
                schema=schema,
                account=account,
                role=role,
                insecure_mode=insecure_mode,
            )

            # === optionally add okta authentication param ===
            if not username.startswith("SVC_"):
                # params["authenticator"] = "https://console.okta.com"
                params["authenticator"] = "externalbrowser"

            # ====== Making snowflake connection ======
            conn = snowflake.connector.connect(**params)

            # ====== Setting warehouse ======
            conn.cursor().execute(
                "USE warehouse {warehouse}".format(warehouse=warehouse)
            )

            # ====== Setting database and schema ======
            conn.cursor().execute(
                "USE {database}.{schema}".format(
                    database=database, schema=schema
                )
            )

            self._conn = conn

    def execute(self, query=None):
        """
        Executes snowflake query.

        Parameters
        ----------
        query : str 
            Snowflake query, either a filepath to a .sql file,
            a str containing a SQL query, or a query object.

        Returns
        -------
        pandas.DataFrame
            Query results.
        """

        # ====== Execution for file inputs =======
        if query.endswith(".sql"):
            with open(query) as file:
                query = file.read()
            # === establishing cursor ===
            cursor = self._conn.cursor()
            # === executing ===
            cursor.execute(query)
            return cursor.fetch_pandas_all()

        # ====== Execution for strings ======
        elif isinstance(query, str):
            cursor = self._conn.cursor()
            cursor.execute(query)
            return cursor.fetch_pandas_all()

        # ====== TypeError if query is non of the above ======
        else:
            raise TypeError(
                "query parameter must be path to a .sql file or a string."
            )


class GroupWithCommandOptions(click.Group):
    """Allow application of options to group with multi command"""

    def add_command(self, cmd, name=None):
        click.Group.add_command(self, cmd, name=name)

        # add the group parameters to the command
        for param in self.params:
            cmd.params.append(param)

        # hook the commands invoke with our own
        cmd.invoke = self.build_command_invoke(cmd.invoke)
        self.invoke_without_command = True

    def build_command_invoke(self, original_invoke):
        def command_invoke(ctx):
            """insert invocation of group function"""

            # separate the group parameters
            ctx.obj = dict(_params=dict())
            for param in self.params:
                name = param.name
                ctx.obj["_params"][name] = ctx.params[name]
                del ctx.params[name]

            # call the group function with its parameters
            params = ctx.params
            ctx.params = ctx.obj["_params"]
            self.invoke(ctx)
            ctx.params = params

            # now call the original invoke (the command)
            original_invoke(ctx)

        return command_invoke