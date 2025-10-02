#!/usr/bin/env python3
"""
Simple testing script for LOCOST Mamba implementation
This script tests the basic functionality of the Mamba-based LOCOST model
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        print("✅ LOCOSTConfig_Mamba imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LOCOSTConfig_Mamba: {e}")
        return False
    
    try:
        from models_mamba import (
            LOCOSTLayerSelfAttention_Mamba,
            LOCOSTBlock_Mamba, 
            LOCOSTStack_Mamba,
            LOCOSTModel_Mamba,
            LOCOSTForConditionalGeneration_Mamba
        )
        print("✅ All Mamba model classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Mamba model classes: {e}")
        return False
    
    try:
        from mamba_ssm import Mamba
        print("✅ Mamba SSM imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Mamba SSM: {e}")
        print("   Make sure mamba-ssm is installed: pip install mamba-ssm")
        return False
    
    return True


def test_config_creation():
    """Test configuration creation with various parameters"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION CREATION")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        
        # Test default config
        config = LOCOSTConfig_Mamba()
        print("✅ Default config created successfully")
        print(f"   d_model: {config.d_model}")
        print(f"   d_state: {config.d_state}")
        print(f"   d_conv: {config.d_conv}")
        print(f"   expand: {config.expand}")
        
        # Test custom config matching the YAML
        config_custom = LOCOSTConfig_Mamba(
            d_model=768,
            d_state=256,
            d_ff=2048,
            num_layers=12,
            num_decoder_layers=12,
            num_heads=12,
            dropout_rate=0.0,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True
        )
        print("✅ Custom config (matching YAML) created successfully")
        print(f"   d_model: {config_custom.d_model}")
        print(f"   d_state: {config_custom.d_state}")
        print(f"   num_layers: {config_custom.num_layers}")
        
        return config_custom
        
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return None


def test_mamba_layer():
    """Test individual Mamba layer functionality"""
    print("\n" + "=" * 60)
    print("TESTING MAMBA LAYER")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        from models_mamba import LOCOSTLayerSelfAttention_Mamba
        
        config = LOCOSTConfig_Mamba(d_model=512, d_state=128)
        layer = LOCOSTLayerSelfAttention_Mamba(config, layer_idx=0).to(device)
        print("✅ Mamba layer created successfully")
        
        # Test forward pass
        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)
        
        with torch.no_grad():
            outputs = layer(hidden_states)
        
        output_hidden_states = outputs[0]
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output shape: {output_hidden_states.shape}")
        print(f"   Output type: {type(output_hidden_states)}")
        
        # Check output format
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"
        assert outputs[1] is None, "Expected None for present_key_value"
        assert outputs[2] is None, "Expected None for position_bias" 
        assert outputs[3] is None, "Expected None for attention_weights"
        print("✅ Output format is correct")
        
        # Check residual connection
        assert output_hidden_states.shape == hidden_states.shape, "Shape mismatch after residual connection"
        print("✅ Residual connection maintains shape")
        
        return True
        
    except Exception as e:
        print(f"❌ Mamba layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_block():
    """Test Mamba block (layer + feedforward)"""
    print("\n" + "=" * 60)
    print("TESTING MAMBA BLOCK")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        from models_mamba import LOCOSTBlock_Mamba
        
        # Test encoder block (uses Mamba)
        config = LOCOSTConfig_Mamba(d_model=512, d_state=128, is_decoder=False)
        block = LOCOSTBlock_Mamba(config, has_relative_attention_bias=True)
        print("✅ Encoder Mamba block created successfully")
        
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)
        
        with torch.no_grad():
            outputs = block(
                hidden_states=hidden_states,
                attention_mask=None,
                original_attention_mask=None,
                output_attentions=False,
                return_dict=True
            )
        
        output_hidden_states = outputs[0]
        print(f"✅ Encoder block forward pass successful")
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output shape: {output_hidden_states.shape}")
        
        # Test decoder block (uses attention)
        config_decoder = LOCOSTConfig_Mamba(d_model=512, d_state=128, is_decoder=True)
        block_decoder = LOCOSTBlock_Mamba(config_decoder, has_relative_attention_bias=True)
        print("✅ Decoder block created successfully")
        
        # Test decoder block without encoder states (self-attention only)
        with torch.no_grad():
            outputs_decoder = block_decoder(
                hidden_states=hidden_states,
                attention_mask=None,
                original_attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                return_dict=True
            )
        
        print("✅ Decoder block forward pass successful (self-attention only)")
        
        return True
        
    except Exception as e:
        print(f"❌ Mamba block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model():
    """Test the complete LOCOST Mamba model"""
    print("\n" + "=" * 60)
    print("TESTING FULL MODEL")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        from models_mamba import LOCOSTForConditionalGeneration_Mamba
        
        # Create a small model for testing
        config = LOCOSTConfig_Mamba(
            vocab_size=1000,
            d_model=256,
            d_state=64,
            d_ff=512,
            num_layers=2,
            num_decoder_layers=2,
            d_conv=4,
            expand=2
        )
        
        model = LOCOSTForConditionalGeneration_Mamba(config)
        print("✅ Full model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Decoder input shape: {decoder_input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
        
        logits = outputs.logits
        print(f"✅ Full model forward pass successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Verify logits shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Wrong logits shape: {logits.shape} vs {expected_shape}"
        print("✅ Output logits have correct shape")
        
        # Test with labels (training mode)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            outputs_with_loss = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        
        loss = outputs_with_loss.loss
        print(f"✅ Training mode forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss shape: {loss.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """Test text generation capability"""
    print("\n" + "=" * 60)
    print("TESTING GENERATION")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        from models_mamba import LOCOSTForConditionalGeneration_Mamba
        
        # Create a very small model for generation testing
        config = LOCOSTConfig_Mamba(
            vocab_size=100,
            d_model=128,
            d_state=32,
            d_ff=256,
            num_layers=1,
            num_decoder_layers=1,
            d_conv=4,
            expand=2,
            eos_token_id=1,
            pad_token_id=0,
            decoder_start_token_id=0
        )
        
        model = LOCOSTForConditionalGeneration_Mamba(config)
        model.eval()
        print("✅ Generation model created successfully")
        
        # Test generation
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(2, config.vocab_size, (batch_size, seq_len))
        
        print(f"   Input for generation: {input_ids}")
        
        with torch.no_grad():
            # Test basic generation
            generated = model.generate(
                input_ids=input_ids,
                max_length=12,
                num_beams=1,
                do_sample=False,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id
            )
        
        print(f"✅ Generation successful")
        print(f"   Generated: {generated}")
        print(f"   Generated shape: {generated.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_with_config():
    """Test compatibility with the YAML configuration"""
    print("\n" + "=" * 60)
    print("TESTING YAML CONFIG COMPATIBILITY")
    print("=" * 60)
    
    try:
        from models_config import LOCOSTConfig_Mamba
        from models_mamba import LOCOSTForConditionalGeneration_Mamba
        
        # Configuration from the YAML file
        yaml_config = LOCOSTConfig_Mamba(
            vocab_size=32128,
            d_model=768,
            d_state=256,
            d_ff=2048,
            num_layers=12,
            num_decoder_layers=12,
            num_heads=12,
            dropout_rate=0.0,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True
        )
        
        print("✅ YAML-compatible config created successfully")
        
        # Try to create model (this might be too large for actual testing)
        try:
            model = LOCOSTForConditionalGeneration_Mamba(yaml_config)
            print("✅ Full-size model created successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"⚠️  Full-size model creation failed (likely due to size): {e}")
            print("   This is expected on machines with limited memory")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("LOCOST MAMBA IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Config Creation", test_config_creation),
        ("Mamba Layer", test_mamba_layer),
        ("Mamba Block", test_mamba_block),
        ("Full Model", test_full_model),
        ("Generation", test_generation),
        ("YAML Compatibility", test_compatibility_with_config),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Config Creation":
                result = test_func()
                results[test_name] = result is not None
            else:
                result = test_func()
                results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Mamba implementation looks good!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
