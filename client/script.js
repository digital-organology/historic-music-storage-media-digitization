var containerHeight = 8500;
var containerWidth  = 8500;
var centerX = -1;
var centerY = -1;
var paper = Raphael("container", containerWidth, containerHeight);

var activeSelection = false;
var selectedList = [];

$("#container").mousedown(function(){
    activeSelection = true;
}).mouseup(function(){
    activeSelection = false;
});


var maxTrackId = 0;
var clickedNotes = new Set();
var currentTrackId = -1;

const CSS_COLOR_NAMES = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
    "#ffffb3"
  ];

// initialise navigation position
var currentX = 0;
var currentY = 0;
var currentS = 8500;


$("#zoomout").on("click",function(){
    currentS = currentS + 0.1 * currentS;
    paper.setViewBox(0, 0, currentS, currentS, true);
    console.log(currentS);
   
});
$("#zoomin").on("click",function(){
    currentS = currentS - 0.1 * currentS;
    $("#container").css("top", "100px");
    paper.setViewBox(0, 0, currentS, currentS, true);
   
});


$("#full").on("click",function(){
    currentS = 39057;
    paper.setViewBox(-200, -50, currentS, currentS, true);
});



function onNoteClick(e) {
    var shapeId = e.target.id;

    if (clickedNotes.has(shapeId)) {
        clickedNotes.delete(shapeId);
        e.target.classList.remove("select");
        e.target.classList.add("deselect");

    } else {
        if (clickedNotes.size == 0) {
        currentTrackId = e.target.dataset.track;
        clickedNotes.add(shapeId);
        
        } else { // check if trackID matches
            if (currentTrackId === e.target.dataset.track) {
                clickedNotes.add(shapeId);

            } else {
                lightNotes(currentTrackId);
                clickedNotes.clear();
                currentTrackId = e.target.dataset.track;
                clickedNotes.add(shapeId);
            }
        }

    e.target.classList.add("select");
    } 

    
    

    $("#presentShapeValue").html(Array.from(clickedNotes).join(', '));
    var color   = CSS_COLOR_NAMES[currentTrackId % CSS_COLOR_NAMES.length];


    $("#presentTrackValue").html(currentTrackId);
    $("#presentTrackColor").css("background-color", color);

    var trackUp   = currentTrackId > 1           ? currentTrackId-1 : currentTrackId;
    var trackDown = currentTrackId < maxTrackId  ? parseInt(currentTrackId)+1 : currentTrackId;

    $("#track1Value").html(trackUp);
    $("#track2Value").html(trackDown);

    $("#track1Color").css("background-color", CSS_COLOR_NAMES[trackUp % CSS_COLOR_NAMES.length]);
    $("#track2Color").css("background-color", CSS_COLOR_NAMES[trackDown % CSS_COLOR_NAMES.length]);

}


function highlightTrack(e) {
    track = e.target.dataset.track;
    trackElements = document.querySelectorAll('[data-track="' + track + '"]');
    trackElements.forEach(node => {
        node.classList.remove("inactive");
        node.classList.add("active");
    })
}

function lowlightTrack(e) {
    track = e.target.dataset.track;
    trackElements = document.querySelectorAll('[data-track="' + track + '"]');
    trackElements.forEach(node => {
        node.classList.remove("active");
        node.classList.add("inactive");
    })
}

function lightNotes(track) {
    trackElements = document.querySelectorAll('[data-track="' + track + '"]');
    trackElements.forEach(node => {
        node.classList.remove("select");
        node.classList.add("deselect");
    })
}


function onSaveClick(e) {
    alert("Save changes");
}

function onDiscardClick(e) {
    alert("Discard changes");
}




fetch("data.json")
    .then(res => res.json())
    .then(data => {
        data["data"].forEach(shape => {
            var note = paper.path(shape.points)
                            .attr({"fill" : CSS_COLOR_NAMES[shape.track % CSS_COLOR_NAMES.length]});
            node_id = "shape_".concat(shape.id);
            note.node.id = node_id;
            note.node.dataset.track = shape.track;
            note.node.dataset.points = shape.points;
            $("#".concat(node_id)).click(onNoteClick);
            $("#".concat(node_id)).on("mouseenter", highlightTrack);
            $("#".concat(node_id)).on("mouseout", lowlightTrack);
            $("#".concat(node_id)).addClass("shape");
            maxTrackId = Math.max(maxTrackId, shape.track);

            centerX = data["center"][0];
            centerY = data["center"][1];
            

        });

        
    $(".shape").mouseenter(function(){
    if(activeSelection)
    {
        if(!selectedList.includes(this.id))
        {
            selectedList.push(this.id);
            console.log(selectedList);
        }
    }
    });

    $("#selectionView").css("top", 200);
    $("#selectionView").css("left", 500);
    $("#selectionView").css("width", 200);
    $("#selectionView").css("height", 80);
    
    $("#track1Color").on("click", function(d) {
        var newTrack = $("#track1Value").html();
        setNewTrackForShape(newTrack);

    });

    $("#track2Color").on("click", function(d) {
        var newTrack = $("#track2Value").html();
        setNewTrackForShape(newTrack);

    });


    function setNewTrackForShape(newTrackId)
    {   
        lightNotes(currentTrackId);
        clickedNotes.forEach(function(shapeId) {
            $("#"+shapeId).attr("fill", CSS_COLOR_NAMES[newTrackId % CSS_COLOR_NAMES.length]);
            $("#"+shapeId).attr("data-track", newTrackId);
        });
        clickedNotes.clear();
    }

    $("#save-btn").on("click", function(d) {
        onSaveClick();
    });

    $("#discard-btn").on("click", function(d) {
        onDiscardClick();
    });

    $("#play-btn").on("click", function(d) {
        var json = {};

        document.querySelectorAll("path").forEach(node => {
            json[node.id.replace("shape_", "")] = node.dataset.track
        });

        fetch("/generate-midi", {
            method: 'POST',
            headers: {
                'Accept': 'text/plain',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(json) 
        })
            .then(data => {
                MIDIjs.play('/tool/flask_midi.mid');
            });
    });


});