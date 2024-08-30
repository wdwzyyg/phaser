
var socket: WebSocket | null = null;

addEventListener("DOMContentLoaded", (event) => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");

    if (!id) {
        socket = null;
        return;
    }
    socket = new WebSocket(`ws://localhost/${id}/listen`);

    socket.addEventListener("open", (event) => {
        console.log("Socket connected");
    });

    socket.addEventListener("error", (event) => {
        console.log("Socket error: ", event);
    });

    socket.addEventListener("message", (event) => {
        console.log("Message from server: ", event.data);
    });

    socket.addEventListener("close", (event) => {
        console.log("Socket disconnected");
    });
});