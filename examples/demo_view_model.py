#!/usr/bin/env python3
"""Demonstration of Ether.view_model() method.

This example shows how to use the view_model method to access model data
without expensive copies, as requested in the user query.
"""

from pydantic import BaseModel

from ether import Ether


@Ether.register(
    payload=["text"],
    metadata=["language"],
    kind="text",
)
class MyModel(BaseModel):
    text: str
    language: str


def main() -> None:
    """Demonstrate view_model usage."""
    print("=== Ether.view_model() Demo ===\n")

    # Create an Ether envelope from a model
    model = MyModel(text="Hello, world!", language="en")
    eth = Ether.from_model(model)

    print(f"Original model: {model}")
    print(f"Ether envelope: {eth}\n")

    # Use view_model to get a lazy view
    mod = eth.view_model(MyModel)

    print("Accessing fields through view (no expensive copies):")
    print(f"mod.text = {mod.text}")
    print(f"mod.language = {mod.language}")

    # Verify it's the same data
    print("\nVerification:")
    print(f"mod.text == model.text: {mod.text == model.text}")
    print(f"mod.language == model.language: {mod.language == model.language}")

    # Convert back to model when needed
    converted_model = mod.as_model()
    print(f"\nConverted back to model: {converted_model}")
    print(f"Converted model type: {type(converted_model)}")
    print(f"Converted model == original model: {converted_model == model}")


if __name__ == "__main__":
    main()
