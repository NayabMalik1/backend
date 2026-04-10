import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter(prefix="/local-report", tags=["Local Report Generation"])

class ScanResultPayload(BaseModel):
    result: Dict[str, Any]

def generate_detailed_report(scan: Dict[str, Any]) -> Dict[str, str]:
    """Generate a rich, human-readable report from the scan data."""
    # Basic info
    label = scan.get("predicted_label", "unknown").upper()
    confidence = scan.get("confidence", 0)
    danger = scan.get("danger_score", 0)
    family = scan.get("predicted_family", "unknown")
    
    # Risk assessment paragraph
    if label == "MALWARE":
        risk = (f"The APK is classified as **malware** with {confidence:.2f}% confidence and a danger score of {danger:.2f}%. "
                f"This indicates a high probability of malicious behaviour. The predicted family is '{family}', "
                f"which typically performs harmful actions such as data theft, SMS fraud, or device control. "
                f"Immediate action is recommended.")
    else:
        risk = (f"The APK is classified as **benign** with {confidence:.2f}% confidence. The danger score is {danger:.2f}%, "
                f"indicating low risk. The predicted family is '{family}', which generally does not exhibit malicious behaviour. "
                f"However, always review requested permissions and API calls for any anomalies.")
    
    # Malware classification paragraph
    if label == "MALWARE":
        classification = (f"The model predicts this APK belongs to the '{family}' malware family with {confidence:.2f}% confidence. "
                          f"Families like '{family}' are known for specific malicious patterns. For instance, banking trojans may steal credentials, "
                          f"while SMS malware sends premium messages. The confidence score suggests the model is "
                          f"{'highly' if confidence > 80 else 'moderately'} certain about this classification.")
    else:
        classification = (f"The APK is classified as benign. The model predicts it belongs to the '{family}' family, "
                          f"which is not associated with known malware behaviour. The confidence score of {confidence:.2f}% indicates "
                          f"{'strong' if confidence > 80 else 'reasonable'} certainty that this app is safe.")
    
    # Permissions analysis
    perms = scan.get("permissions", [])
    if perms:
        perm_list = "\n".join(perms)
        perm_analysis = f"The APK requests the following permissions:\n{perm_list}\n\n"
        dangerous = [p for p in perms if any(d in p for d in ["SEND_SMS", "READ_SMS", "READ_CONTACTS", "ACCESS_FINE_LOCATION", "CAMERA", "RECORD_AUDIO"])]
        if dangerous:
            perm_analysis += f"Dangerous permissions detected: {', '.join(dangerous)}. These can lead to privacy breaches or financial loss."
        else:
            perm_analysis += "No highly dangerous permissions (SMS, contacts, location, camera, microphone) are requested."
    else:
        perm_analysis = "No permissions were extracted from the APK. This is unusual for a typical Android app; the file may be corrupted or not a standard APK."
    
    # API calls analysis
    apis = scan.get("api_calls", [])
    if apis:
        api_list = "\n".join(apis)
        api_analysis = f"The APK contains the following suspicious API calls:\n{api_list}\n\n"
        if any("SmsManager" in a for a in apis):
            api_analysis += "`SmsManager` calls can send SMS messages without user interaction, leading to premium charges or spam."
        if any("HttpURLConnection" in a for a in apis):
            api_analysis += "`HttpURLConnection` indicates network activity – possible data exfiltration or command‑and‑control communication."
        if any("Runtime.exec" in a for a in apis):
            api_analysis += "`Runtime.exec` allows executing arbitrary commands, which is highly suspicious and could lead to system compromise."
    else:
        api_analysis = "No suspicious API calls were detected."
    
    # Behaviour patterns (derived from permissions)
    patterns = []
    if any("SEND_SMS" in p for p in perms):
        patterns.append("Sends SMS")
    if any("READ_SMS" in p for p in perms):
        patterns.append("Reads SMS")
    if any("READ_CONTACTS" in p for p in perms):
        patterns.append("Reads Contacts")
    if any("ACCESS_FINE_LOCATION" in p for p in perms):
        patterns.append("Accesses Location")
    if any("CAMERA" in p for p in perms):
        patterns.append("Uses Camera")
    if any("RECORD_AUDIO" in p for p in perms):
        patterns.append("Records Audio")
    if any("INTERNET" in p for p in perms):
        patterns.append("Internet Access")
    behavior = f"Behaviour patterns: {', '.join(patterns) if patterns else 'None'}."
    
    # Network activity
    network = "HTTP network calls detected – potential data exfiltration." if "HttpURLConnection" in str(apis) else "No suspicious network indicators."
    
    # Permission‑behaviour correlation
    correlation = ("The APK requests permissions that align with its observed behaviour patterns. "
                   "For example, if it requests `READ_CONTACTS` and uses `HttpURLConnection`, it could send contact data to a remote server. "
                   "Review the combination of permissions and API calls to assess risk.") if perms and apis else "Insufficient data for correlation analysis."
    
    # Data leakage
    data_leakage = ("Potential data leakage points include: sending SMS messages, reading contacts, accessing location, and network communication. "
                    "If the app transmits this data to external servers, it may violate user privacy.") if any(p in perms for p in ["SEND_SMS", "READ_CONTACTS", "ACCESS_FINE_LOCATION"]) else "No obvious data leakage indicators."
    
    # Security vulnerabilities
    vulnerabilities = ("The presence of `Runtime.exec` or dynamic code loading could introduce command injection vulnerabilities. "
                      "Additionally, using `HttpURLConnection` without proper certificate validation may expose data to man‑in‑the‑middle attacks.") if any("Runtime.exec" in a for a in apis) else "No high‑risk vulnerabilities detected from the static analysis."
    
    # Behaviour timeline
    timeline = ("Upon installation, the app could request sensitive permissions, then in the background use those permissions to collect data and send it over the network. "
                "If SMS permissions are granted, it might silently send premium messages. A timeline would require dynamic analysis to confirm actual behaviour.")
    
    # IOCs
    iocs = "Indicators of compromise include the combination of dangerous permissions (e.g., `SEND_SMS`, `READ_CONTACTS`) and network API calls. "
    if apis:
        iocs += f"Suspicious API calls: {', '.join(apis[:3])}."
    else:
        iocs += "No specific API‑based IOCs found."
    
    # Final verdict
    verdict = ("⚠️ **Malicious – block installation.**" if label == "MALWARE" else "✅ **Benign – safe to use.**")
    
    return {
        "risk_assessment": risk,
        "malware_classification": classification,
        "permissions_analysis": perm_analysis,
        "api_call_analysis": api_analysis,
        "code_behavior_analysis": behavior,
        "network_activity": network,
        "permission_behavior_correlation": correlation,
        "data_leakage_analysis": data_leakage,
        "security_vulnerabilities": vulnerabilities,
        "behavior_timeline": timeline,
        "indicators_of_compromise": iocs,
        "final_verdict": verdict
    }

@router.post("/generate")
async def generate_local_report(payload: ScanResultPayload):
    # Always return the detailed template (fast and reliable)
    return generate_detailed_report(payload.result)