import { Component, OnInit } from '@angular/core';
import { StrapiService } from './services/strapi.service';

@Component({
  selector: 'app-root',
  template: `
    <h1>Articles</h1>
    <div *ngFor="let article of articles">
      <h2>{{ article.attributes.title }}</h2>
      <p>{{ article.attributes.content }}</p>
    </div>
  `,
  styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit {
  articles: any[] = [];

  constructor(private strapiService: StrapiService) {}

  ngOnInit(): void {
    this.strapiService.getArticles().subscribe({
      next: (data) => {
        this.articles = data.data; // Strapi returns data under the 'data' field
      },
      error: (err) => {
        console.error('Error fetching articles:', err);
      },
    });
  }
}
