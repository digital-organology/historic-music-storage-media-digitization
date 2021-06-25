//$('#main-recent-filter').toggle();
//$('#main-recent-filter').draggable();

// Creates canvas 320 × 200 at 10, 50
var containerHeight = 8500;
var containerWidth  = 8500;
var centerX = -1;
var centerY = -1;
var paper = Raphael("container", containerWidth, containerHeight);
// paper.setViewBox(-200, -50);
// paper.setViewBox(0, 0, 5000, 5000, true);
// paper.setSize('100%', '100%');

// var selectionViewCoords = [];
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
    
    if (clickedNotes.size == 0) {
        currentTrackId = e.target.dataset.track;
        clickedNotes.add(e.target.id);
        
    } else { // check if trackID matches
        if (currentTrackId === e.target.dataset.track) {
            clickedNotes.add(e.target.id);

        } else {
            lowlightNote(currentTrackId);
            clickedNotes.clear();

            currentTrackId = e.target.dataset.track;
            clickedNotes.add(e.target.id);

        }
    }
    e.target.classList.add("select");
    

    

    // var trackId = e.target.dataset.track;
    // var noteId  = e.target.id;
    // var color   = CSS_COLOR_NAMES[trackId % CSS_COLOR_NAMES.length];
    // $("#presentShapeValue").html(noteId);
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

//    alert("Note ID: ".concat(e.target.id).concat("; Track ID: ").concat(e.target.dataset.track));
}

// function highlightNote(e) {
//     note = e.target.id;

// }

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

function lowlightNote(track) {
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
            maxTrackId = Math.max(maxTrackId, shape.track);

            centerX = data["center"][0];
            centerY = data["center"][1];
            
           // $("#main-recent-filter").css("top", centerY - parseInt($("#main-recent-filter").css("height")));
            //$("#main-recent-filter").css("left", centerX - parseInt($("#main-recent-filter").css("width")));

        });
        // selectionViewCoords = data["center"];

    // $("body").append("<div id='selectionView' class='selectionView'></div>");
    $("#selectionView").css("top", 200);//selectionViewCoords[1]-selectionViewCoords[2]/2);
    $("#selectionView").css("left", 500);//, selectionViewCoords[0]-selectionViewCoords[2]/2);
    $("#selectionView").css("width", 200);//, selectionViewCoords[2]);
    $("#selectionView").css("height", 80);//, selectionViewCoords[2]);
    
    $("#track1Color").on("click", function(d) {
        var newTrack = $("#track1Value").html();
        // var shapeId  = $("#presentShapeValue").html();
        // setNewTrackForShape(shapeId, newTrack);
        setNewTrackForShape(newTrack);

    });

    $("#track2Color").on("click", function(d) {
        var newTrack = $("#track2Value").html();
        // var shapeId  = $("#presentShapeValue").html();
        // setNewTrackForShape(shapeId, newTrack);
        setNewTrackForShape(newTrack);

    });


    // function setNewTrackForShape(shapeId, newTrackId)
    function setNewTrackForShape(newTrackId)
    {   
        lowlightNote(currentTrackId);
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

});