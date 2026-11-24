export async function triggerModelLoad() {
  const res = await fetch("/api/load", { method: "POST" });
  return res.json();
}

export async function startBrandJob(payload) {
  const res = await fetch("/api/brand/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Failed to start generation.");
  return data;
}

export async function cancelBrandJob() {
  const res = await fetch("/api/brand/cancel", { method: "POST" });
  return res.json();
}
