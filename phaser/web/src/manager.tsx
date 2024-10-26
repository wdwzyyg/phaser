
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, createStore, Provider } from 'jotai';

import { Reconstruction } from './types';
import { Section } from './components';


let socket: WebSocket | null = null;
const reconstructions: PrimitiveAtom<Array<Reconstruction>> = atom([] as Array<Reconstruction>);
const store = createStore();

function start_recons(e: React.MouseEvent) {
    fetch("/start", {
        method: "POST",
        body: "",
    })
    .then((response) => response.json())
    .then((json) => {
        //window.location = json['links']['dashboard'];
        console.log(`Got response: ${JSON.stringify(json)}`);
    });
};

function cancel_recons(recons: Reconstruction, e: React.MouseEvent) {
    console.log(`cancelRecons: recons: ${JSON.stringify(recons)}`);
    fetch(recons.links.cancel, {
        method: "POST",
        body: "",
    })
    .then((response) => response.json())
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
};

export function Reconstructions(props: {}) {
    const cols = [20, 60, 10, 10].map((w, i) => <col style={{width: `${w}%`}} key={i}></col>);
    const headers = ["ID", "Status", "Watch", "Cancel"].map((name, i) => <th scope="col" key={i}>{name}</th>);

    const recons = useAtomValue(reconstructions);

    const rows = recons.map((recons) => {
        return <tr key={recons.id}>
            <td>{recons.id}</td>
            <td>{recons.state}</td>
            <td><a className="simple-button" href={recons.links.dashboard}></a></td>
            <td><button className="simple-button" onClick={(e) => cancel_recons(recons, e)}></button></td>
        </tr>;
    });

    return <table>
        <colgroup>{cols}</colgroup>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows}</tbody>
    </table>;
}

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <Provider store={store}>
            <Section name="Start reconstruction">
                <button onClick={start_recons}>Start reconstruction</button>
            </Section>
            <Section name="Current reconstruction">
                <Reconstructions/>
            </Section>
        </Provider>
    </StrictMode>
);

addEventListener("DOMContentLoaded", (event) => {
    socket = new WebSocket(`ws://${window.location.host}/listen`);

    socket.addEventListener("open", (event) => {
        console.log("Socket connected");
    });

    socket.addEventListener("error", (event) => {
        console.log("Socket error: ", event);
    });

    socket.addEventListener("message", (event) => {
        console.log(`Socket event: ${event.data}`)
        let data = JSON.parse(event.data);
        store.set(reconstructions, (prev) => data.state);
    });

    socket.addEventListener("close", (event) => {
        console.log("Socket disconnected");
    });

    let sections = document.getElementsByClassName("section-header");
    for (let i = 0; i < sections.length; i++) {
        const sibling = sections[i].nextElementSibling;
        if (sibling === null || !sibling?.classList.contains("section")) {
            continue
        }
        sections[i].addEventListener("click", function() {
            sibling.classList.toggle("collapsed");
        })
    }
});