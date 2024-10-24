
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import Section from './section';

import { PlotScale } from './plotting/scale';
import { Figure, Plot, PlotGrid, AxisSpec, Axis } from './plotting/plot';

const axes: Map<string, AxisSpec> = new Map([
    ["x1", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "X1",
        show: 'one',
    }],
    ["x2", {
        scale: new PlotScale([-4.0, 4.0], [0.0, 400.0]),
        label: "X2",
        show: 'one',
    }],
    ["y1", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "Y1",
        show: 'one',
    }],
    ["y2", {
        scale: new PlotScale([-2.0, 2.0], [0.0, 200.0]),
        label: "Y2",
        show: 'one',
    }],
]);

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <Section name="Section 1">
            <Figure axes={axes}>
                <Plot xaxis="x1" yaxis="y1">
                    <rect x="50" y="50" width="100" height="100" />
                </Plot>
            </Figure>
        </Section>
        <Section name="Section 2">
            <Figure axes={axes}>
                <PlotGrid ncols={2} nrows={2} xaxes={"x1"} yaxes={"y1"}>
                    <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                    <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                    <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                    <Plot fixedAspect={true}><rect x="50" y="50" width="100" height="100" /></Plot>
                </PlotGrid>
            </Figure>
        </Section>
    </StrictMode>
);

/*
import * as d3 from 'd3';

var reconstructions: Array<any> = [];
var socket: WebSocket | null = null;


function initTable() {
    let table = d3.select("#recons-table");

    let colgroup = table.append("colgroup");
    colgroup.append("col").attr("style", "width: 20%");
    colgroup.append("col").attr("style", "width: 60%");
    colgroup.append("col").attr("style", "width: 10%");
    colgroup.append("col").attr("style", "width: 10%");

    let header = table.append("thead").append("tr");
    header.append("th").attr("scope", "col").html("ID");
    header.append("th").attr("scope", "col").html("Status");
    header.append("th").attr("scope", "col").html("Watch");
    header.append("th").attr("scope", "col").html("Cancel");

    table.append("tbody");
}

function updateTable() {
    let table = d3.select("#recons-table > tbody");

    let rows = table.selectAll("tr")
        .data(reconstructions)
        .join(function(enter) {
            let row = enter.append("tr");

            row.append("td").html((d) => d.id);
            row.append("td").html((d) => "Running");
            row.append("td").append("a")
                .attr("class", "simple-button")
                .attr("href", (d) => d.links.dashboard);
            row.append("td").append("button")
                .attr("class", "simple-button")
                .on("click", cancelRecons);

            return row;
        })

    rows.each((v, i, groups) => console.log(`row ${i}: ${v} ${groups}`));
}

function watchRecons(this: HTMLElement, event: MouseEvent, datum: any) {
    console.log(`watchRecons: ${JSON.stringify(event)} this: ${this}, datum: ${JSON.stringify(datum)}`);
    window.location = datum.links.watch;
}

function cancelRecons(this: HTMLElement, event: MouseEvent, datum: any) {
    console.log(`cancelRecons: ${JSON.stringify(event)} this: ${this}, datum: ${JSON.stringify(datum)}`);
    fetch(datum.links.cancel, {
        method: "POST",
        body: "",
    })
    .then((response) => response.json())
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
}

addEventListener("DOMContentLoaded", (event) => {
    initTable();

    document.getElementById('start')!.onclick = (event) => {
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

    var i = 0

    document.getElementById('start-fake')!.onclick = (event) => {
        reconstructions.push({'id': `id${i}`, 'status': "running"})
        i += 1;
        updateTable();
    }

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
        reconstructions = data.state;
        updateTable();
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
*/