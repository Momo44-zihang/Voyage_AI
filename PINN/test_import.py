import sys
import os

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print("Testing imports...")
print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("\n1. Importing tensorflow...")
    import tensorflow as tf
    print(f"   OK - TensorFlow {tf.__version__}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n2. Importing PINN modules...")
    from PINN.src.models import tov_pinn
    print("   OK - tov_pinn")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Creating model...")
    model = tov_pinn.TOV_PINN_with_IC()
    print("   OK - Model created")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")

