"""Send one short message to the model and print the reply and token usage.

A smoke test that the key, client and model name work. Costs a fraction of a cent.
"""

import argparse

from config import DEFAULT_MODEL, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Ping the Anthropic API once.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    client = get_client()
    response = client.messages.create(
        model=args.model,
        max_tokens=256,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Reply with one short sentence: are you there?"}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    print(f"Model:       {response.model}")
    print(f"Stop reason: {response.stop_reason}")
    print(f"Reply:       {text}")
    print(f"Tokens:      input={response.usage.input_tokens}, output={response.usage.output_tokens}")


if __name__ == "__main__":
    main()
