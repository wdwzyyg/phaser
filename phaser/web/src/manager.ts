
addEventListener("DOMContentLoaded", (event) => {
    document.getElementById('start')!.onclick = (event) => {
        fetch("/start", {
            method: "POST",
            body: "",
        })
        .then((response) => response.json())
        .then((json) => {
            window.location = json['links']['dashboard'];
        });
    };
});
