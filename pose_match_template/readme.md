
# Pose Match
## Invite visitors to explore your open access GLAM collection through poses

![screenshot of Pose Match with a user on the left side of the screen and a matched image on the right. The user has reference points marked with green dots on their eyes and shoulders, with red lines connecting the dots](/readme_images/pose-match-screenshot.png)

Welcome to Pose Match, a project from the [GLAM-E Lab](https://www.glamelab.org/).  This repo will guide you through the process of setting up Pose Match using your own open access collection.  This can be done for free using publicly available resources.  If you want to get started, jump down to the [Getting Started](#getting-started) section.  If you want more background on the project, read on below.

The process involves three simple steps:

1. Prepare your images with the included python scripts
2. Test everything locally
3. Deploy your version to Cloudflare Pages to make it available online for free

<table>
  <tr>
    <td><a href="https://www.glamelab.org/"><img src="/readme_images/logo-vertical.png" alt="GLAM-E Lab" width="200"></a></td>
    <td><a href="https://www.nyuengelberg.org/"><img src="/readme_images/ec-lockup-color.png" alt="Engelberg Center logo" width="200"></a></td>
  </tr>
</table>


# Background

## What is Pose Match?

Pose Match is a different way for visitors to explore your open access collection.  

Visitors who come to collections with a specific interest in mind ("Dutch Masters" or "photographs of birds that live in the desert") are usually well served by existing presentations. Pose Match was designed for users who have a generalized curiosity about a collection and need an entry point to get started on their exploration.

## What does Pose Match do?

The premise of Pose Match is simple - a user strikes a pose and is matched to an item in the collection with a similar pose.

![screenshot of Pose Match with a user on the left side of the screen and a matched image on the right. The user has reference points marked with green dots on their eyes and shoulders, with red lines connecting the dots](/readme_images/pose-match-screenshot.png)

A user can click on the matched image or scan the QR code to find out more information about the matched image.

![screenshot of more information screen with an image of the matched work above metadata including name, artist, date medium, department, and repository](/readme_images/more-info-screenshot.png%20.png)

Depending on the metadata you have in your collection, this page can include information about the work, where it can be found in your institution, as well as other information.

## How can I use Pose Match?

Pose Match is browser based.  That means you can make it available as part of your online collection, or install it as a physical kiosk in your institution.

## Privacy
>[!NOTE]
>All image processing for Pose Match happens locally. User images are not saved or sent to any third parties.  In fact, you can run a basic version of Pose Match without any internet connection at all (although you will need some sort of server to allow users to visit the object pages for more information). 

## Does Pose Match use AI?

Yes, although the full answer to that question depends on how you define "use" and how you define "AI".

Pose Match uses computer vision to identify poses.  Specifically, it uses [ml5.js BodyPose](https://docs.ml5js.org/#/reference/bodypose), which is built on TensorFlow's MoveNet and BlazePose models.  This allows Pose Match to identify key points (nose, left eye, right knee, etc.) in the open access works and the user.

Pose Match is also developed with coding assistance from LLMs. As of this writing that mostly includes Claude Code.

Since Pose Match does not send any data to third parties, none of the Pose Match user data is sent to AI companies for training. 


# Getting Started

This repo is designed to allow you to clone and deploy as quickly as possible.  The [Deployment](#deployment) section will get you started.

## Overview

This overview section explains some of the architecture and elements of Pose Match. 

### Architecture and Workflow Overview
<details>
<summary>An overview of the Pose Match architecture</summary>

The simplest version of Pose Match runs as a website on a publicly accessible server.  That server includes a collection of open access images and a json file that contains metadata about the images.  The metadata includes traditional metadata that your institution has about objects in its collection (title, artist, medium, etc.).  It also contains metadata about the poses represented in the work. You will create that metadata before you deploy Pose Match using the scripts in this repo.

When a user engages with Pose Match, the ml5.js library identifies key points of the user (nose, left eye, right knee, etc.), draws those points on the screen, and uses cosine distance to calculate a mathematical relationship between the points.  It then searches the json file for an object with points that closely match that mathematical relationship (the smallest cosine distance).  This match is displayed next to the live video image of the user.

Clicking on the matching image or scanning the QR code sends the user to an `/info` page.  The url for that page includes an identifier for the work (for example, `https://pose-match.pages.dev/info?id=SSsaam_1973.164` includes the id `SSsaam_1973.164`).  When a user visits `/info?id=IDENTIFIER`, the identifier is passed to the page and matched to the json file.  The info page then loads the relevant metadata onto the info page.  

This `/info` page structure means that the main Pose Match page and the `/info` page can operate independently.  They do not talk to each other behind the scenes.  `/info` will load metadata for any object contained in its json file in response to a valid identifier.  Users can interact with a deployment of Pose Match that is completely offline (for example, in the lobby of a museum). The QR code can take them to the `/info` page hosted elsewhere.  

This model works because deployment effectively happens in two stages.  During the **preparation phase** you will assemble images and metadata, create the mathematical representations of the poses, and finalize the json of metadata.  During the **deployment phase** you will make your version of Pose Match available online.

</details>

### Ways to Deploy

Pose Match can be deployed online or in your institution's physical location.  In both cases, users will interact with a webcam and monitor attached to a computer (see [minimum requirements](#minimum-requirements)).

You can make your version of Pose Match available for free using a combination of github and cloudflare pages.  

### Minimum Requirements

Pose Match is lightweight to operate and runs in the browser.  It will run on any modern laptop, as well as a inexpensive computer such as a raspberry pi.  You will need:

1. Webcam (no meaningful minimum for the resolution)
2. Monitor
3. Computer (client will run on a Raspberry Pi)
4. A server running Pose Match (may be the same as the computer running the user interaction)
5. Metadata about the objects in your collection
6. Images of objects in your collection


## Deployment

Before deploying, you need to gather metadata about the objects in your collection and images of objects in your collection.  The pose detection does not require the images to be high resolution, so pick image sizes that can be loaded quickly on a website and make sense for your intended monitor size.  If you are planning on using Cloudflare Pages to deploy your version of Pose Match, each image file must be less than 25 MiB. 

> [!TIP]
>This template already contains an example `input_image_metadata.json` and `/input_images` folder.  You can use these to test that it is running properly before preparing your own images (see the [Test Locally](#step-3-test-locally) section below).  You will want to delete these before importing your own images. 

### Preparation Phase

You will use a series of python scripts to prepare and import your data. They can be found in the `/utilities` folder. 

#### Step 1: /image_cleaner

`person_or_not.py` is the utility to take a bunch of arbitrary images, determine if they contain people, and save metadata including pose information.  It uses tensorflow to identify keypoints for poses and cosine distances to describe the relationship between those points.  Before running, you may need to install some libraries (depending on your platform you may want to do this in a virtual environment):

```bash
pip install numpy tensorflow tensorflow-hub Pillow
```

`person_or_not.py` must be run from inside the `utilities/image_cleaner` folder.  

`person_or_not.py` assumes:
* Your images are in a folder called `input_images` in the `image_cleaner` folder,
* That the images are formatted as JPEG files,
* That the filename for each of those images is `objectID.jpg`. The `objectID` can be arbitrary as long as it matches the metadata schema you are using in the metadata json.
* That there is metadata about the image files in a file called `input_metadata.json` in the `image_cleaner` folder that includes the fields:
    * `Object_ID`
    * `Title`
    * `Artist_Display_Name`
    * `Object_Date`
    * `Department`
    * `Medium`
    * `Link_Resource`	
    * `Repository`

`Object_ID` should match the filename convention you use for the files in the input_images folder. So `Object_1234.jpg` will correspond to an `Object_ID` value of `Object_1234`.
`Link_Resource` is a link to the resource in an online collection.
`Repository` is the name of the institution (or portion of the institution) that is the source of the object.  

Any field can be left blank.

`person_or_not.py` outputs:
* A new copy of any images that probably have people into `/processed_images`
* metadata, including pose information, for those images into `image_metadata.json`

#### Step 2: image_cleaner_to_main.py

`image_cleaner_to_main.py` is a utility to copy the output of `person_or_not.py` into the main project directory. You could do this manually, but `image_cleaner_to_main.py` was created to avoid common errors that manual copying often introduced.

`image_cleaner_to_main.py` must be run from inside the `utilities` folder.  If you just ran `person_or_not.py` from inside the `utilities/image_cleaner` folder be sure to change your directory.

This script runs in two modes:

* **replace mode** will replace the contents of the main project folder  (/input_images and input_image_metadata.json) with the contents of the image_cleaner utility
* **append mode** will add the contents of the image_cleaner utility to the existing contents of the main project folder.

Set the mode on line 16 of the script.

When you run the script it will prompt you for a 2 character source code.  This code will be added to the file identifier and name.  The purpose of the source code is to avoid namespace collisions.  For example, if you import an image called `12345.jpg` from one collection and a different `12345.jpg` from a different collection, the unique character code appended to the different collections will mean that instead of two `12345.jpg` files, you have `AA12345.jpg` and `BB12345.jpg`.

If you are just importing one set of images, it does not really matter what you choose. However, if you are importing multiple set of images, pick a unique code that will help you identify the source of each set. This can also be helpful if you ever need to track down the source of a file directly.  

Once this is complete, you will have a folder called `input_images` in your main directory that is full of your images and a file called `input_image_metadata.json` with metadata about those images.

#### Step 3: Test Locally

At this point, everything should work when you test it on your local machine. You will need to start a local server to run the code ([instructions here](https://github.com/processing/p5.js/wiki/Local-server)).  Opening up index.html should show you the home page. You will be prompted to allow the page to access the webcam (that prompt will vary depending on your browser).  Grant permission and things should start running.  If not, raise an issue in this repo and we will do our best to help you sort things out.  

If everything looks good, you are ready to make the page available *on the internet*.

### Deployment Phase

Pose Match is self contained, so there are many ways you could host it. This section describes a way to host your Pose Match for free.

The approach is to connect a github repo hosting your version of Pose Match to Cloudflare pages.  Cloudflare pages will pull the information from the repo and make it available at a specific URL. By default that URL will be [REPO NAME].pages.dev. However, it is also fairly straightforward to host it at a custom domain, or a subdomain for your existing site (like posematch.example.com).

You will need a free github account and a free Cloudflare account.  Once you have those, [create a new github repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository) and then [connect it to Cloudflare pages](https://developers.cloudflare.com/pages/configuration/git-integration/).  That's it!


## Further Customization

The `index.html`, `info.html`, and `about.html` are all self contained so you can edit the text on each of them directly.  `info.html` will only display metadata that exists in `input_image_metadata.json` so it should not cause problems if you data does not contain values for individual fields.

In addition, you can make specific customizations:

**branding.js** contains editable values for colors and fonts.  Changing those variables will change the appearance of all three pages.

**analytics.js** contains places to add your analytics snippets.  They will then be copied to all three pages.  Note that Pose Match does not natively include cookie banners.

**/branding** contains `favicon.png` and `logo.svg`.  You can replace these files with your preferred favicon and logo.  It also contains a `/fonts` folder.  You can add your preferred fonts, and update references to them in `branding.js`. 


# Credits and Thanks

Huge thanks to the [ml5.js](https://ml5js.org/) team for creating such a user friendly way to work with powerful machine learning.  It makes it much easier to create projects that used to be hard.

The Google Creative Lab documented [their own version of this project in 2018](https://medium.com/tensorflow/move-mirror-an-ai-experiment-with-pose-estimation-in-the-browser-using-tensorflow-js-2f7b769f9b23).  That post helped explain how to connect the ability to identify a pose with actually matching an existing work.

Finally, as we started developing this project, many (many, many) people responded with a variation of "oh like [Strike a Pose at the Cleveland Museum of Art](https://www.clevelandart.org/artlens-gallery/artlens-gallery-first-iteration-gallery-one)?".  The only answer to that is "yes, a lot like that!".  

