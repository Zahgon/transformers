
from huggingface_hub import check_cli_update, typer_factory

from transformers.cli.add_new_model_like import add_new_model_like
from transformers.cli.chat import Chat
from transformers.cli.download import download
from transformers.cli.serve import Serve
from transformers.cli.system import env, version


app = typer_factory(help="Transformers CLI")

app.command()(add_new_model_like)
app.command(name="chat")(Chat)
app.command()(download)
app.command()(env)
app.command(name="serve")(Serve)
app.command()(version)


def main():
    check_cli_update("transformers")
    app()


if __name__ == "__main__":
    main()
