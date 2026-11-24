// brand-generator.js
// API layer: talks to the server, never touches the DOM.
// TODO 1: send a request to "/api/load"
export async function triggerModelLoad() {
  
  const res = await fetch("/api/load", { method: "POST"});
  const response = await res.json();
  return response;
}

export async function startBrandJob(payload) {
  const res = await fetch("/api/brand/start", { method: "POST" , headers: { "Content-type": "application/json; charset=UTF-8"}, body: JSON.stringify(payload)} ); 
  const response = await res.json();

  if (!res.ok) {
      throw new Error(response.error ||"Image could not be generated");
  }
  return response; 
}

export async function cancelBrandJob() {
  const res = await fetch("/api/brand/cancel", { method: "POST"}); 
  const response = await res.json();
  return response;
}