"""
Inference script demonstrating the complete InnerLight system
"""

import torch
from src import inference, rag


def run_inference_demo(classifier_path, generator_path, rag_artifact_path):
    """
    Run complete InnerLight inference demo.
    
    Args:
        classifier_path (str): Path to trained classifier
        generator_path (str): Path to trained generator
        rag_artifact_path (str): Path to RAG artifacts
    """
    print("\n" + "="*70)
    print("INNERLIGHT INFERENCE DEMO")
    print("="*70)
    
    # Load models for inference
    print("\nLoading models...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    
    # Load classifier
    clf_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    clf_model, clf_pipe = inference.load_classifier_for_inference(
        "bert-base-uncased",
        classifier_path,
        clf_tokenizer
    )
    print("✓ Classifier loaded")
    
    # Load generator
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
    gen_model, gen_pipe = inference.load_generator_for_inference(
        "Qwen/Qwen1.5-1.8B",
        generator_path,
        gen_tokenizer
    )
    print("✓ Generator loaded")
    
    # Load RAG artifacts
    rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(rag_artifact_path)
    print("✓ RAG system loaded")
    
    # Test cases
    test_cases = [
        ("I'm having trouble sleeping lately and feeling stressed about work.", "US"),
        ("I feel so hopeless and alone. Nothing seems to matter anymore.", "US"),
        ("I've been thinking about suicide. I don't see a way out of this pain.", "United States"),
        ("Can you recommend some relaxation techniques for anxiety?", "US"),
        ("Everything is falling apart and I can't take it anymore.", "Canada"),
    ]
    
    print("\n" + "="*70)
    print("Running inference tests...")
    print("="*70)
    
    for i, (user_input, location) in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test Case {i}")
        print(f"{'─'*70}")
        print(f"User Input: {user_input}\n")
        
        # Run inference with RAG
        result = rag.process_with_rag(
            user_input,
            clf_pipe,
            gen_pipe,
            rag_index,
            rag_model,
            rag_texts,
            rag_data,
            location
        )
        
        # Display output
        output = rag.format_rag_output(result)
        print(output)
    
    print("\n" + "="*70)
    print("Inference demo complete!")
    print("="*70)


def interactive_mode(classifier_path, generator_path, rag_artifact_path):
    """
    Run interactive inference mode.
    
    Args:
        classifier_path (str): Path to trained classifier
        generator_path (str): Path to trained generator
        rag_artifact_path (str): Path to RAG artifacts
    """
    print("\n" + "="*70)
    print("INNERLIGHT INTERACTIVE MODE")
    print("="*70)
    print("\nLoading models (this may take a moment)...")
    
    from transformers import AutoTokenizer
    
    # Load models
    clf_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    clf_model, clf_pipe = inference.load_classifier_for_inference(
        "bert-base-uncased",
        classifier_path,
        clf_tokenizer
    )
    
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
    gen_model, gen_pipe = inference.load_generator_for_inference(
        "Qwen/Qwen1.5-1.8B",
        generator_path,
        gen_tokenizer
    )
    
    # Load RAG artifacts
    rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(rag_artifact_path)
    
    print("✓ All models loaded!\n")
    print("You can now chat with the InnerLight system.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using InnerLight. Stay safe!")
            break
        
        if not user_input:
            continue
        
        location = input("Your location (or press Enter for US): ").strip() or "US"
        
        # Process input
        result = rag.process_with_rag(
            user_input,
            clf_pipe,
            gen_pipe,
            rag_index,
            rag_model,
            rag_texts,
            rag_data,
            location
        )
        
        # Display output
        output = rag.format_rag_output(result)
        print(output)
        print()


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="InnerLight Inference Script")
    parser.add_argument("--classifier_path", type=str, required=True,
                       help="Path to trained classifier")
    parser.add_argument("--generator_path", type=str, required=True,
                       help="Path to trained generator")
    parser.add_argument("--rag_path", type=str, required=True,
                       help="Path to RAG artifacts")
    parser.add_argument("--mode", type=str, choices=['demo', 'interactive'],
                       default='demo',
                       help="Inference mode (demo or interactive)")
    
    args = parser.parse_args()
    
    # Verify paths exist
    for path_name, path in [
        ('classifier', args.classifier_path),
        ('generator', args.generator_path),
        ('rag', args.rag_path)
    ]:
        if not os.path.exists(path):
            print(f"Error: {path_name} path does not exist: {path}")
            exit(1)
    
    if args.mode == 'demo':
        run_inference_demo(args.classifier_path, args.generator_path, args.rag_path)
    else:
        interactive_mode(args.classifier_path, args.generator_path, args.rag_path)
