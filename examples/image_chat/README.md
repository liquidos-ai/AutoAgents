# Image Chat Example

This example demonstrates how to use AutoAgents with image messages using OpenAI's vision-capable models.

## Prerequisites

You'll need an OpenAI API key and a test image file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Using the default test image

If you run the example without specifying an image, it will use the default test image (`test_img.jpg`) included in the example directory:

```bash
cargo run -p image-chat-example
```

### Specifying an image

You can specify your own image file using the `--image` or `-i` flag:

```bash
cargo run -p image-chat-example -- --image /path/to/your/image.jpg
```
