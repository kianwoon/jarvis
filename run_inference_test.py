import asyncio
import sys
from app.llm.test_inference import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        sys.exit(1) 