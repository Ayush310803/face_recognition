from fastapi import Request, HTTPException

ALLOWED_IPS = {"127.0.0.1", "192.168.1.10", "203.0.113.5"}

async def check_ip(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="Access forbidden: IP not allowed")
    return await call_next(request)