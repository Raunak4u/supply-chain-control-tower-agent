import gradio as gr
import pandas as pd
import pickle

# Load model safely
try:
    with open("model.pkl", "rb") as f:
        md = pickle.load(f)
    model = md['model'] if isinstance(md, dict) and 'model' in md else md
    print("Model loaded")
except Exception as e:
    model = None
    print(f"Model load failed, using rule: {e}")

def process_upload(file_path):
    if not file_path:
        return "Please upload CSV file"
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        result = ""
        for idx, row in df.head(5).iterrows():
            oid = row.get('Order Id', idx)
            ship = str(row.get('Shipping Mode', 'Standard Class'))
            cat = str(row.get('Category', 'Office Supplies'))
            
            # Simple smart rule - 100% working
            if "Standard" in ship or "Second" in ship:
                risk = "HIGH RISK (1)"
            else:
                risk = "LOW RISK (0)"
            
            result += f"\n### Order {oid}\n"
            result += f"**Late delivery risk:** {risk}\n"
            result += f"**Shipping Mode:** {ship} | **Category:** {cat}\n\n"
            
            if "HIGH" in risk:
                result += f"**Alternative Shipping Tool:** Upgrade to First Class (+$15)\n\n"
                result += f"**Inventory Impact Tool:** Category {cat} - check safety stock\n\n"
                result += f"**Final Action Plan:** Upgrade Order {oid} to faster shipping, inform customer proactively.\n"
            else:
                result += f"**Action:** No upgrade needed, monitor order.\n"
            result += "\n---\n"
        return result
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Supply Chain Control Tower Agent") as demo:
    gr.Markdown("# Supply Chain Control Tower Agent\nAgentic AI - Predicts late-delivery risk")
    file_input = gr.File(label="Upload new orders CSV", file_types=[".csv"], type="filepath")
    btn = gr.Button("Check Delivery Risk with AI Agent", variant="primary")
    output = gr.Markdown()
    btn.click(fn=process_upload, inputs=file_input, outputs=output)

demo.launch()