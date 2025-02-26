import argparse
import unittest
import sys

# Import the utils module. Depending on your project structure, you might need to adjust this import.
from huggingfaceserver.vllm import utils


# Define a dummy AsyncEngineArgs to simulate the behavior for testing
class DummyAsyncEngineArgs:
    def __init__(self):
        self.speculative_model = None
        self.speculative_model_revision = None
        self.num_speculative_tokens = None

    @classmethod
    def add_cli_args(cls, parser):
        # For testing, just return the parser unchanged
        return parser

    @classmethod
    def from_cli_args(cls, args):
        # Return an instance and let the caller set attributes
        return cls()


class TestSpeculativeModelArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Monkey-patch the AsyncEngineArgs in the utils module
        utils.AsyncEngineArgs = DummyAsyncEngineArgs
        # Ensure _vllm is set to True so that our functions add the CLI arguments
        utils._vllm = True

    def test_default_arguments(self):
        # Test that if no speculative model arguments are provided, they remain None
        parser = argparse.ArgumentParser()
        parser = utils.maybe_add_vllm_cli_parser(parser)

        # Simulate empty command-line arguments
        args = parser.parse_args([])
        engine_args = utils.build_vllm_engine_args(args)

        self.assertIsNone(engine_args.speculative_model, "Expected speculative_model to be None by default")
        self.assertIsNone(engine_args.speculative_model_revision, "Expected speculative_model_revision to be None by default")
        self.assertIsNone(engine_args.num_speculative_tokens, "Expected num_speculative_tokens to be None by default")

    def test_with_speculative_model_arguments(self):
        # Test that providing speculative model arguments sets them in engine args
        test_model = "path/to/speculative/model"
        test_revision = "v1.0"
        test_num_tokens = 10
        parser = argparse.ArgumentParser()
        parser = utils.maybe_add_vllm_cli_parser(parser)

        # Simulate providing command-line arguments
        args = parser.parse_args([
            "--speculative-model", test_model,
            "--speculative-model-revision", test_revision,
            "--num-speculative-tokens", str(test_num_tokens)
        ])
        engine_args = utils.build_vllm_engine_args(args)

        self.assertEqual(engine_args.speculative_model, test_model, "speculative_model should match the given value")
        self.assertEqual(engine_args.speculative_model_revision, test_revision, "speculative_model_revision should match the given value")
        self.assertEqual(engine_args.num_speculative_tokens, test_num_tokens, "num_speculative_tokens should match the given value")


if __name__ == '__main__':
    unittest.main()
